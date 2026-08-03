/*
    Copyright (C) 2026  Holy Wu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

// Converted to the VapourSynth GPU node API: frames stay resident on the
// core's Vulkan device (clip:vnode:gpu in and out), all Vulkan calls go through
// the core's dispatch table, and the former CPU stages — building the padded
// field plane and interleaving the interpolated rows into the output frame —
// run on the GPU (pad.comp and buffer copies recorded into the same
// submission). Nothing is uploaded or downloaded per frame anymore; the
// filter chains asynchronously through per-plane producer pairs like any
// other GPU filter. Scratch memory comes from the core's pooled VRAM
// allocator, so it is budgeted and recycled centrally.
//
// Relative to the standalone version this drops: the private instance/device
// (volk + VMA), the device_index/list_device arguments (device selection is
// core-wide now, core.set_vulkan_device), the readback machinery, and the
// VK_NV_cooperative_vector predict path (the core device enables no vendor
// extensions).

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define VS_USE_API_43
#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

using namespace std::string_literals;

// Must match common.glsl.
constexpr int MARGIN_H = 24;
constexpr int MARGIN_V = 3;

const uint32_t padU8Spv[] = {
#include "pad_u8.h"
};
const uint32_t padU16Spv[] = {
#include "pad_u16.h"
};
const uint32_t padF16Spv[] = {
#include "pad_f16.h"
};
const uint32_t padF32Spv[] = {
#include "pad_f32.h"
};
const uint32_t prescreenU8Spv[] = {
#include "prescreen_u8.h"
};
const uint32_t prescreenU16Spv[] = {
#include "prescreen_u16.h"
};
const uint32_t prescreenF16Spv[] = {
#include "prescreen_f16.h"
};
const uint32_t prescreenF32Spv[] = {
#include "prescreen_f32.h"
};
const uint32_t predictU8Spv[] = {
#include "predict_u8.h"
};
const uint32_t predictU16Spv[] = {
#include "predict_u16.h"
};
const uint32_t predictF16Spv[] = {
#include "predict_f16.h"
};
const uint32_t predictF32Spv[] = {
#include "predict_f32.h"
};
// Byte addressed, so one variant serves every pixel type.
const uint32_t rowcopySpv[] = {
#include "rowcopy.h"
};

struct SpvBlob {
    const uint32_t* code;
    size_t size;
};

// Indexed [kernel][pixelType]; kernels: 0 = pad, 1 = prescreen, 2 = predict.
constexpr SpvBlob kSpv[3][4] = {
    { { padU8Spv, sizeof(padU8Spv) },
      { padU16Spv, sizeof(padU16Spv) },
      { padF16Spv, sizeof(padF16Spv) },
      { padF32Spv, sizeof(padF32Spv) } },
    { { prescreenU8Spv, sizeof(prescreenU8Spv) },
      { prescreenU16Spv, sizeof(prescreenU16Spv) },
      { prescreenF16Spv, sizeof(prescreenF16Spv) },
      { prescreenF32Spv, sizeof(prescreenF32Spv) } },
    { { predictU8Spv, sizeof(predictU8Spv) },
      { predictU16Spv, sizeof(predictU16Spv) },
      { predictF16Spv, sizeof(predictF16Spv) },
      { predictF32Spv, sizeof(predictF32Spv) } },
};

#define VK_CHECK(expr)                                                                             \
    do {                                                                                           \
        const VkResult vkCheckResult_ = (expr);                                                    \
        if (vkCheckResult_ != VK_SUCCESS)                                                          \
            throw std::runtime_error(#expr " failed with VkResult "s +                             \
                                     std::to_string(static_cast<int>(vkCheckResult_)));            \
    } while (0)

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) & ~(a - 1);
}

// Sub-allocation alignment within the shared pad/device buffers (the spec
// caps minStorageBufferOffsetAlignment at 256).
constexpr VkDeviceSize BUF_ALIGN = 256;

// Reserves bytes at the next aligned offset and returns that offset.
VkDeviceSize suballoc(VkDeviceSize& off, VkDeviceSize bytes) {
    off = alignUp(off, BUF_ALIGN);
    const VkDeviceSize o = off;
    off += bytes;
    return o;
}

bool envFlag(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] && env[0] != '0';
}

// Round-to-nearest-even float32 -> float16 conversion.
uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7fffffffu;

    if (absx >= 0x7f800000u) // inf/nan
        return static_cast<uint16_t>(sign | 0x7c00u | ((absx > 0x7f800000u) ? 0x200u : 0u));
    if (absx >= 0x477ff000u) // overflows to inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (absx < 0x38800000u) { // subnormal or zero
        if (absx < 0x33000001u)
            return static_cast<uint16_t>(sign);
        const int shift = 125 - static_cast<int>(absx >> 23);
        const uint32_t mant = (absx & 0x7fffffu) | 0x800000u;
        uint32_t half = mant >> (shift + 1);
        const uint32_t rem = mant & ((2u << shift) - 1);
        const uint32_t halfway = 1u << shift;
        if (rem > halfway || (rem == halfway && (half & 1u)))
            half++;
        return static_cast<uint16_t>(sign | half);
    }

    uint32_t half = ((absx >> 13) & 0x3ffu) | ((((absx >> 23) - 112u) & 0x1fu) << 10);
    const uint32_t rem = absx & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
        half++;
    return static_cast<uint16_t>(sign | half);
}

//////////////////////////////////////////
// NNEDI3 weights (ported from znedi3)

constexpr size_t NNEDI3_WEIGHTS_SIZE = 13574928;
constexpr unsigned NNEDI3_XDIM[] = { 8, 16, 32, 48, 8, 16, 32 };
constexpr unsigned NNEDI3_YDIM[] = { 6, 6, 6, 6, 4, 4, 4 };
constexpr unsigned NNEDI3_NNS[] = { 16, 32, 64, 128, 256 };

struct PrescreenerOldCoefficients {
    float kernel_l0[4][12 * 4];
    float bias_l0[4];

    float kernel_l1[4][4];
    float bias_l1[4];

    float kernel_l2[4][8];
    float bias_l2[4];
};

struct PrescreenerNewCoefficients {
    float kernel_l0[4][16 * 4];
    float bias_l0[4];

    float kernel_l1[4][4];
    float bias_l1[4];
};

struct PredictorModel {
    unsigned xdim = 0, ydim = 0, nns = 0;
    std::vector<float> softmax_q1, elliott_q1, softmax_bias_q1, elliott_bias_q1;
    std::vector<float> softmax_q2, elliott_q2, softmax_bias_q2, elliott_bias_q2;
};

double vecMean(const float* buf, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++)
        acc += buf[i];
    return acc / n;
}

template<typename Coeffs>
void subtractMean(Coeffs& coeffs, double pixelHalf) {
    for (unsigned n = 0; n < 4; n++) {
        const double m = vecMean(coeffs.kernel_l0[n], std::size(coeffs.kernel_l0[n]));
        for (float& x : coeffs.kernel_l0[n])
            x = static_cast<float>((x - m) / pixelHalf);
    }
}

void subtractMean(PredictorModel& model) {
    const size_t filterSize = model.xdim * model.ydim;
    const unsigned nns = model.nns;

    std::vector<double> softmaxMeans(nns);       // average of individual softmax filters
    std::vector<double> elliottMeans(nns);       // average of individual elliott filters
    std::vector<double> meanFilter(filterSize);  // pointwise average of all softmax filters
    double meanBias;

    const auto onePass = [&](float* softmax, float* elliott, float* softmaxBias) {
        std::fill(meanFilter.begin(), meanFilter.end(), 0.0);

        for (unsigned nn = 0; nn < nns; nn++) {
            softmaxMeans[nn] = vecMean(softmax + nn * filterSize, filterSize);
            elliottMeans[nn] = vecMean(elliott + nn * filterSize, filterSize);

            for (size_t k = 0; k < filterSize; k++)
                meanFilter[k] += softmax[nn * filterSize + k] - softmaxMeans[nn];
        }
        for (size_t k = 0; k < filterSize; k++)
            meanFilter[k] /= nns;
        meanBias = vecMean(softmaxBias, nns);

        for (unsigned nn = 0; nn < nns; nn++) {
            for (size_t k = 0; k < filterSize; k++) {
                softmax[nn * filterSize + k] -= static_cast<float>(softmaxMeans[nn] + meanFilter[k]);
                elliott[nn * filterSize + k] -= static_cast<float>(elliottMeans[nn]);
            }
            softmaxBias[nn] -= static_cast<float>(meanBias);
        }
    };

    onePass(model.softmax_q1.data(), model.elliott_q1.data(), model.softmax_bias_q1.data());
    onePass(model.softmax_q2.data(), model.elliott_q2.data(), model.softmax_bias_q2.data());
}

// Walks the nnedi3 weights file exactly like znedi3's read_nnedi3_weights and
// extracts the prescreeners plus the single predictor model that was asked
// for (etypeSel: 0 = abs, 1 = mse).
void readNNEDI3Weights(const float* data, unsigned nsizeSel, unsigned nnsSel, unsigned etypeSel,
                       PrescreenerOldCoefficients& psOld, PrescreenerNewCoefficients psNew[3],
                       PredictorModel& model) {
    const float* ptr = data;
    auto read = [&](float* dst, size_t n) {
        std::copy_n(ptr, n, dst);
        ptr += n;
    };

    // Old prescreener data.
    read(&psOld.kernel_l0[0][0], 4 * 48);
    read(psOld.bias_l0, 4);
    read(&psOld.kernel_l1[0][0], 4 * 4);
    read(psOld.bias_l1, 4);
    read(&psOld.kernel_l2[0][0], 4 * 8);
    read(psOld.bias_l2, 4);

    // New prescreener data.
    for (unsigned i = 0; i < 3; i++) {
        float kernelL0Shuffled[4 * 64];
        float kernelL1Shuffled[4 * 4];

        read(kernelL0Shuffled, 4 * 64);
        read(psNew[i].bias_l0, 4);
        read(kernelL1Shuffled, 4 * 4);
        read(psNew[i].bias_l1, 4);

        // Convert kernels back to row-major order.
        for (unsigned n = 0; n < 4; n++) {
            for (unsigned k = 0; k < 64; k++)
                psNew[i].kernel_l0[n][k] = kernelL0Shuffled[(k / 8) * 32 + n * 8 + k % 8];
            for (unsigned k = 0; k < 4; k++)
                psNew[i].kernel_l1[n][k] = kernelL1Shuffled[k * 4 + n];
        }
    }

    // ABS models, then MSE models; grouped by neuron count, then window size.
    for (unsigned m = 0; m < 2; m++) {
        for (unsigned i = 0; i < 5; i++) {
            for (unsigned j = 0; j < 7; j++) {
                const unsigned nns = NNEDI3_NNS[i];
                const size_t filterSize = NNEDI3_XDIM[j] * NNEDI3_YDIM[j];

                if (m == etypeSel && i == nnsSel && j == nsizeSel) {
                    model.xdim = NNEDI3_XDIM[j];
                    model.ydim = NNEDI3_YDIM[j];
                    model.nns = nns;

                    model.softmax_q1.resize(nns * filterSize);
                    model.elliott_q1.resize(nns * filterSize);
                    model.softmax_bias_q1.resize(nns);
                    model.elliott_bias_q1.resize(nns);
                    model.softmax_q2.resize(nns * filterSize);
                    model.elliott_q2.resize(nns * filterSize);
                    model.softmax_bias_q2.resize(nns);
                    model.elliott_bias_q2.resize(nns);

                    read(model.softmax_q1.data(), nns * filterSize);
                    read(model.elliott_q1.data(), nns * filterSize);
                    read(model.softmax_bias_q1.data(), nns);
                    read(model.elliott_bias_q1.data(), nns);
                    read(model.softmax_q2.data(), nns * filterSize);
                    read(model.elliott_q2.data(), nns * filterSize);
                    read(model.softmax_bias_q2.data(), nns);
                    read(model.elliott_bias_q2.data(), nns);
                } else {
                    ptr += 4 * nns * filterSize + 4 * nns;
                }
            }
        }
    }

    assert(static_cast<size_t>(ptr - data) == NNEDI3_WEIGHTS_SIZE / sizeof(float));
}

//////////////////////////////////////////
// Filter data

// Field order and types must match the PC push-constant block in
// shaders/common.glsl (scalar layout makes the correspondence purely
// positional). srcStride/rowOff/rowScale/fp only matter to the pad and
// rowcopy kernels; dstBase/dstPitch place the strided output rows.
struct PushConstants {
    int32_t width;
    int32_t rows;
    int32_t padStride;
    int32_t peak;
    int32_t srcStride;
    int32_t rowOff;
    int32_t rowScale;
    int32_t fp;
    int32_t dstBase;
    int32_t dstPitch;
};

static_assert(sizeof(PushConstants) == 10 * 4, "must stay in sync with the PC block in common.glsl");

// Field order must match the constant_id assignments in the shaders (see
// specEntries).
struct SpecData {
    uint32_t wgSize;
    int32_t pscrn;
    int32_t xdim;
    int32_t ydim;
    int32_t nns;
    int32_t qual;
    int32_t sgSize;
    int32_t subgroups; // pixels per predict workgroup
    VkBool32 useList;
};

// Storage-buffer binding indices; must match the layout(binding = N)
// declarations across the kernels. The pad kernel reuses BindPad as its
// source plane and BindDst as the padded output.
enum Binding : uint32_t {
    BindPad = 0,
    BindDst,
    BindPsW,
    BindPdW,
    BindPdB,
    BindList,
    BindCnt,
    BindCount,
};

struct PlaneSetup {
    bool process = false;
    int width = 0, height = 0; // output plane dimensions
    int rows = 0;              // interpolated rows (= field height)
    int padStride = 0, padHeight = 0;
    VkDeviceSize padOffset = 0, padBytes = 0;   // pad scratch buffer
    VkDeviceSize listOffset = 0, listBytes = 0; // device scratch buffer
    VkDeviceSize cntOffset = 0;                 // device scratch buffer, 16 bytes
};

struct NNEDI3Data {
    VSNode* node = nullptr;
    VSVideoInfo vi{};
    int field = 0;
    bool dh = false;
    int qual = 1, pscrn = 2;
    int peak = 0, pixelType = 0;

    const VSVULKANAPI* vkapi = nullptr;
    VSVulkanCoreHandles h{};
    const VSVulkanFunctions* vk = nullptr; // the core's dispatch table
    VkQueue computeQueue = VK_NULL_HANDLE;

    uint32_t subgroupSize = 32;
    bool canRequireSubgroupSize = false;
    double timestampPeriod = 0.0;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule padModule = VK_NULL_HANDLE, rowcopyModule = VK_NULL_HANDLE;
    VkShaderModule prescreenModule = VK_NULL_HANDLE, predictModule = VK_NULL_HANDLE;
    VkPipeline padPipe = VK_NULL_HANDLE, rowcopyPipe = VK_NULL_HANDLE;
    VkPipeline prescreenPipe = VK_NULL_HANDLE, predictPipe = VK_NULL_HANDLE;

    VSGPUBuffer* weightsBuf = nullptr;
    VSVulkanBufferInfo weightsInfo{};
    VkDeviceSize psWOffset = 0, psWBytes = 0;
    VkDeviceSize pdWOffset = 0, pdWBytes = 0;
    VkDeviceSize pdBOffset = 0, pdBBytes = 0;

    PlaneSetup planes[3];
    VkDeviceSize padSize = 0, devbufSize = 0;

    uint32_t prescreenWG = 128;      // prescreen workgroup size (threads)
    uint32_t pixelsPerPredictWG = 4; // predict: pixels per workgroup

    // The core's exec pool owns the timeline, the command buffers and the in
    // flight bound (num_streams contexts), and it keeps source frames and
    // scratch alive until the submissions using them complete.
    VSGPUExecPool* execPool = nullptr;

    bool profile = false;
    std::mutex profileMutex;
    double prescreenMs = 0.0, predictMs = 0.0, copyMs = 0.0;
    int64_t profiledFrames = 0;

    void destroy() {
        if (!vk)
            return;
        const VkDevice device = h.device;
        // Drains the GPU and releases every frame and scratch buffer still
        // held by an unfinished submission.
        if (execPool)
            vkapi->freeGPUExecPool(execPool);
        execPool = nullptr;
        if (padPipe)
            vk->vkDestroyPipeline(device, padPipe, nullptr);
        if (rowcopyPipe)
            vk->vkDestroyPipeline(device, rowcopyPipe, nullptr);
        if (prescreenPipe)
            vk->vkDestroyPipeline(device, prescreenPipe, nullptr);
        if (predictPipe)
            vk->vkDestroyPipeline(device, predictPipe, nullptr);
        if (padModule)
            vk->vkDestroyShaderModule(device, padModule, nullptr);
        if (rowcopyModule)
            vk->vkDestroyShaderModule(device, rowcopyModule, nullptr);
        if (prescreenModule)
            vk->vkDestroyShaderModule(device, prescreenModule, nullptr);
        if (predictModule)
            vk->vkDestroyShaderModule(device, predictModule, nullptr);
        if (pipelineLayout)
            vk->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (dsl)
            vk->vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        if (weightsBuf)
            vkapi->destroyGPUBuffer(weightsBuf);

        if (profile && profiledFrames > 0)
            std::fprintf(stderr,
                         "nnedi3vk profile: frames=%lld pad+prescreen=%.3fms predict=%.3fms assemble=%.3fms (per frame GPU time)\n",
                         static_cast<long long>(profiledFrames), prescreenMs / profiledFrames,
                         predictMs / profiledFrames, copyMs / profiledFrames);
    }
};

//////////////////////////////////////////
// GPU recording

void cmdMemoryBarrier(const VSVulkanFunctions* vk, VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage,
                      VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    const VkMemoryBarrier2 mb{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                               .srcStageMask = srcStage,
                               .srcAccessMask = srcAccess,
                               .dstStageMask = dstStage,
                               .dstAccessMask = dstAccess };
    const VkDependencyInfo dep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .memoryBarrierCount = 1,
                                .pMemoryBarriers = &mb };
    vk->vkCmdPipelineBarrier2(cmd, &dep);
}

//////////////////////////////////////////
// getFrame

const VSFrame* VS_CC nnedi3GetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core,
                                    const VSAPI* vsapi) {
    auto d = static_cast<NNEDI3Data*>(instanceData);

    // Source frame number (field > 1 doubles the output frame rate).
    const int sn = d->field > 1 ? n / 2 : n;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
        return nullptr;
    }

    if (activationReason != arAllFramesReady)
        return nullptr;

    const VSFrame* src = vsapi->getFrameFilter(sn, d->node, frameCtx);

    // Processed planes are written fresh on the GPU; unprocessed planes are
    // shared, producer pairs riding along.
    VSFrame* dst;
    {
        bool anyShared = false;
        const VSFrame* planeSrc[3] = {};
        int planeNo[3] = {};
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            planeSrc[plane] = (d->dh || d->planes[plane].process) ? nullptr : src;
            planeNo[plane] = plane;
            anyShared = anyShared || planeSrc[plane];
        }
        if (anyShared)
            dst = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, planeSrc, planeNo, src, core);
        else
            dst = d->vkapi->newGPUVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    }

    // Source field parity, mirroring vsznedi3's get_src_parity exactly.
    // parity == 1 means the source is (treated as) the bottom field.
    const VSMap* srcProps = vsapi->getFramePropertiesRO(src);
    const int defaultParity = (d->field == 0 || d->field == 2) ? 1 : 0;
    int parity;
    int err;
    if (d->dh) {
        const int fieldProp = vsapi->mapGetIntSaturated(srcProps, "_Field", 0, &err);
        parity = err ? defaultParity : fieldProp;
    } else if (d->field > 1) {
        const int fieldBased = vsapi->mapGetIntSaturated(srcProps, "_FieldBased", 0, &err);
        parity = fieldBased == VSC_FIELD_BOTTOM ? 1 : fieldBased == VSC_FIELD_TOP ? 0 : defaultParity;
        if (n % 2)
            parity = !parity;
    } else {
        parity = d->field == 0 ? 1 : 0;
    }
    parity = !!parity;

    const int fp = !parity; // znedi3 PadFilter parity
    const int bps = d->vi.format.bytesPerSample;

    // Claim a recording slot; the pool waits out the oldest submission, which
    // is this instance's in flight bound.
    char verr[512] = { 0 };
    VSGPUExecContext* ctx = d->vkapi->gpuExecAcquire(d->execPool, verr, sizeof(verr));
    if (!ctx) {
        vsapi->setFilterError(("NNEDI3VK: "s + verr).c_str(), frameCtx);
        vsapi->freeFrame(src);
        vsapi->freeFrame(dst);
        return nullptr;
    }
    const VkCommandBuffer cmd = d->vkapi->gpuExecCommandBuffer(ctx);
    VSVulkanBufferInfo padInfo{}, devInfo{};

    try {
        // Per frame scratch out of the core's pooled allocator: the exec pool
        // destroys both once this submission completes, so the recycled block
        // is back in the bucket by the next frame.
        VSGPUBuffer* padBuf = d->vkapi->createGPUBuffer(core, d->padSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &padInfo, verr, sizeof(verr));
        if (!padBuf)
            throw std::runtime_error("pad scratch buffer: "s + verr);
        d->vkapi->gpuExecUsesBuffer(ctx, padBuf);

        VSGPUBuffer* devBuf = d->vkapi->createGPUBuffer(core, d->devbufSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &devInfo, verr, sizeof(verr));
        if (!devBuf)
            throw std::runtime_error("device scratch buffer: "s + verr);
        d->vkapi->gpuExecUsesBuffer(ctx, devBuf);

        const int numPlanes = d->vi.format.numPlanes;

        VSVulkanPlaneInfo srcPlanes[3]{}, dstPlanes[3]{};
        for (int plane = 0; plane < numPlanes; plane++) {
            if (!d->planes[plane].process)
                continue;
            if (d->vkapi->getGPUPlane(src, plane, &srcPlanes[plane]) ||
                d->vkapi->getGPUPlane(dst, plane, &dstPlanes[plane]))
                throw std::runtime_error("plane is not GPU resident");
        }

        auto pushPC = [&](const PushConstants& pcv) {
            const VkPushConstantsInfo pi{ .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
                                          .layout = d->pipelineLayout,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .size = sizeof(pcv),
                                          .pValues = &pcv };
            d->vk->vkCmdPushConstants2(cmd, &pi);
        };

        auto pushDescriptors = [&](const VkDescriptorBufferInfo* bufs, uint32_t count) {
            VkWriteDescriptorSet writes[BindCount];
            for (uint32_t i = 0; i < count; i++)
                writes[i] = VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                  .dstBinding = i,
                                                  .descriptorCount = 1,
                                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                  .pBufferInfo = &bufs[i] };
            d->vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipelineLayout, 0, count, writes);
        };

        // Pass-through rows: the source field's rows land unchanged at output
        // rows parity + 2r (or every source row when dh). Byte addressed, so
        // every push constant of the rowcopy kernel is in bytes.
        d->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->rowcopyPipe);
        for (int plane = 0; plane < numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;
            const int srcStride = static_cast<int>(vsapi->getStride(src, plane));
            const int dstStride = static_cast<int>(vsapi->getStride(dst, plane));
            const int rowBytes = p.width * bps;
            const VkDescriptorBufferInfo bufs[2] = {
                { srcPlanes[plane].buffer, 0, VK_WHOLE_SIZE },
                { dstPlanes[plane].buffer, 0, VK_WHOLE_SIZE },
            };
            pushDescriptors(bufs, 2);
            pushPC(PushConstants{ .width = rowBytes,
                                  .rows = p.rows,
                                  .padStride = 0,
                                  .peak = 0,
                                  .srcStride = 0,
                                  .rowOff = d->dh ? 0 : parity * srcStride,
                                  .rowScale = srcStride * (d->dh ? 1 : 2),
                                  .fp = 0,
                                  .dstBase = parity * dstStride,
                                  .dstPitch = dstStride * 2 });
            d->vk->vkCmdDispatch(cmd, (static_cast<uint32_t>(rowBytes) + 63) / 64,
                                 (static_cast<uint32_t>(p.rows) + 3) / 4, 1);
        }

        // Pad: source plane -> padded field plane in the pad scratch.
        d->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->padPipe);
        for (int plane = 0; plane < numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;
            const VkDescriptorBufferInfo bufs[2] = {
                { srcPlanes[plane].buffer, 0, VK_WHOLE_SIZE },
                { padInfo.buffer, p.padOffset, p.padBytes },
            };
            pushDescriptors(bufs, 2);
            pushPC(PushConstants{ .width = p.width,
                                  .rows = p.rows,
                                  .padStride = p.padStride,
                                  .peak = 0,
                                  .srcStride = static_cast<int>(vsapi->getStride(src, plane)) / bps,
                                  .rowOff = d->dh ? 0 : parity,
                                  .rowScale = d->dh ? 1 : 2,
                                  .fp = fp,
                                  .dstBase = 0,
                                  .dstPitch = 0 });
            d->vk->vkCmdDispatch(cmd, (static_cast<uint32_t>(p.padStride) + 15) / 16,
                                 (static_cast<uint32_t>(p.padHeight) + 15) / 16, 1);
        }

        // Reset the per-plane counters and indirect dispatch arguments.
        if (d->pscrn > 0) {
            for (int plane = 0; plane < numPlanes; plane++) {
                const PlaneSetup& p = d->planes[plane];
                if (!p.process)
                    continue;
                d->vk->vkCmdFillBuffer(cmd, devInfo.buffer, p.cntOffset, 8, 0);     // predCount, groupsX
                d->vk->vkCmdFillBuffer(cmd, devInfo.buffer, p.cntOffset + 8, 8, 1); // groupsY, groupsZ
            }
        }

        // Pad writes and counter fills become visible to the network kernels.
        cmdMemoryBarrier(d->vk, cmd,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        auto pushKernelDescriptors = [&](int plane) {
            const PlaneSetup& p = d->planes[plane];
            const bool hasList = d->pscrn > 0;
            VkDescriptorBufferInfo bufs[BindCount];
            bufs[BindPad] = { padInfo.buffer, p.padOffset, p.padBytes };
            bufs[BindDst] = { dstPlanes[plane].buffer, 0, VK_WHOLE_SIZE };
            bufs[BindPsW] = { d->weightsInfo.buffer, d->psWOffset, d->psWBytes };
            bufs[BindPdW] = { d->weightsInfo.buffer, d->pdWOffset, d->pdWBytes };
            bufs[BindPdB] = { d->weightsInfo.buffer, d->pdBOffset, d->pdBBytes };
            bufs[BindList] = { devInfo.buffer, hasList ? p.listOffset : 0, hasList ? p.listBytes : VK_WHOLE_SIZE };
            bufs[BindCnt] = { devInfo.buffer, hasList ? p.cntOffset : 0, hasList ? 16 : VK_WHOLE_SIZE };
            pushDescriptors(bufs, BindCount);
        };

        // Interpolated rows go straight into the destination plane, strided:
        // row r of the field lands at output row !parity + 2r.
        auto kernelPC = [&](int plane) {
            const PlaneSetup& p = d->planes[plane];
            const int dstStrideElems = static_cast<int>(vsapi->getStride(dst, plane)) / bps;
            pushPC(PushConstants{ .width = p.width,
                                  .rows = p.rows,
                                  .padStride = p.padStride,
                                  .peak = d->peak,
                                  .srcStride = 0,
                                  .rowOff = 0,
                                  .rowScale = 0,
                                  .fp = 0,
                                  .dstBase = !parity * dstStrideElems,
                                  .dstPitch = 2 * dstStrideElems });
        };

        if (d->pscrn > 0) {
            d->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->prescreenPipe);
            for (int plane = 0; plane < numPlanes; plane++) {
                const PlaneSetup& p = d->planes[plane];
                if (!p.process)
                    continue;
                pushKernelDescriptors(plane);
                kernelPC(plane);
                const int pixPerThread = (d->pscrn == 1) ? 1 : 4;
                const uint32_t threads = static_cast<uint32_t>(p.rows) *
                    ((static_cast<uint32_t>(p.width) + pixPerThread - 1) / pixPerThread);
                d->vk->vkCmdDispatch(cmd, (threads + d->prescreenWG - 1) / d->prescreenWG, 1, 1);
            }
        }

        if (d->pscrn > 0)
            cmdMemoryBarrier(d->vk, cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

        d->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->predictPipe);
        for (int plane = 0; plane < numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;
            pushKernelDescriptors(plane);
            kernelPC(plane);
            if (d->pscrn > 0) {
                d->vk->vkCmdDispatchIndirect(cmd, devInfo.buffer, p.cntOffset + 4);
            } else {
                const uint32_t pixels = static_cast<uint32_t>(p.width) * static_cast<uint32_t>(p.rows);
                d->vk->vkCmdDispatch(cmd, (pixels + d->pixelsPerPredictWG - 1) / d->pixelsPerPredictWG, 1, 1);
            }
        }

        // The source planes' producers become device side waits and the frame
        // stays alive until this submission completes; the planes written get
        // this submission's producer pair published on them at submit.
        d->vkapi->gpuExecReadsFrame(ctx, src);
        for (int plane = 0; plane < numPlanes; plane++)
            if (d->planes[plane].process)
                d->vkapi->gpuExecWritesPlane(ctx, dst, plane);

        if (d->vkapi->gpuExecSubmit(ctx, verr, sizeof(verr))) {
            ctx = nullptr; // consumed either way
            throw std::runtime_error(verr);
        }
        ctx = nullptr;
    } catch (const std::exception& e) {
        if (ctx)
            d->vkapi->gpuExecAbandon(ctx);
        vsapi->setFilterError(("NNEDI3VK: "s + e.what()).c_str(), frameCtx);
        vsapi->freeFrame(src);
        vsapi->freeFrame(dst);
        return nullptr;
    }

    // The pool holds its own reference for as long as the GPU needs the frame,
    // so this one is released the ordinary way.
    vsapi->freeFrame(src);

    VSMap* props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
    vsapi->mapDeleteKey(props, "_Field");

    if (d->field > 1) {
        int errNum, errDen;
        int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
        int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, 1, 2);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }
    }

    return dst;
}

void VS_CC nnedi3Free(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<NNEDI3Data*>(instanceData);
    d->destroy();
    vsapi->freeNode(d->node);
    delete d;
}

//////////////////////////////////////////
// Creation / Vulkan object setup

// Uploads the prepared weight blobs into a device-local buffer via a one-shot
// staging copy on the core's compute queue.
void uploadWeights(NNEDI3Data* d, VSCore* core, const void* data, VkDeviceSize bytes) {
    char verr[512] = { 0 };

    d->weightsBuf = d->vkapi->createGPUBuffer(core, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &d->weightsInfo, verr, sizeof(verr));
    if (!d->weightsBuf)
        throw std::runtime_error("weights buffer: "s + verr);

    VSVulkanBufferInfo stagingInfo{};
    VSGPUBuffer* staging = d->vkapi->createGPUBuffer(core, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &stagingInfo, verr, sizeof(verr));
    if (!staging)
        throw std::runtime_error("weights staging buffer: "s + verr);

    try {
        std::memcpy(stagingInfo.mapped, data, bytes);

        // One shot: record the copy in a pool context, submit, and drain, since
        // everything recorded afterwards reads these weights.
        VSGPUExecContext* ctx = d->vkapi->gpuExecAcquire(d->execPool, verr, sizeof(verr));
        if (!ctx)
            throw std::runtime_error("weights upload: "s + verr);
        const VkBufferCopy2 region{ .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .size = bytes };
        const VkCopyBufferInfo2 copy{ .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                      .srcBuffer = stagingInfo.buffer,
                                      .dstBuffer = d->weightsInfo.buffer,
                                      .regionCount = 1,
                                      .pRegions = &region };
        d->vk->vkCmdCopyBuffer2(d->vkapi->gpuExecCommandBuffer(ctx), &copy);
        if (d->vkapi->gpuExecSubmit(ctx, verr, sizeof(verr)))
            throw std::runtime_error("weights upload: "s + verr);
        if (d->vkapi->gpuExecPoolWaitIdle(d->execPool, verr, sizeof(verr)))
            throw std::runtime_error("weights upload: "s + verr);
    } catch (...) {
        d->vkapi->destroyGPUBuffer(staging);
        throw;
    }

    d->vkapi->destroyGPUBuffer(staging);
}

void setupVulkanObjects(NNEDI3Data* d, int32_t xdim, int32_t ydim, int32_t nns,
                        uint32_t maxWGInvocations, uint32_t maxWGSizeX) {
    const VkDevice device = d->h.device;
    const int pixelType = d->pixelType;

    {
        const VkShaderModuleCreateInfo smci{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                             .codeSize = kSpv[0][pixelType].size,
                                             .pCode = kSpv[0][pixelType].code };
        VK_CHECK(d->vk->vkCreateShaderModule(device, &smci, nullptr, &d->padModule));
    }
    {
        const VkShaderModuleCreateInfo smci{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                             .codeSize = sizeof(rowcopySpv),
                                             .pCode = rowcopySpv };
        VK_CHECK(d->vk->vkCreateShaderModule(device, &smci, nullptr, &d->rowcopyModule));
    }
    if (d->pscrn > 0) {
        const VkShaderModuleCreateInfo smci{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                             .codeSize = kSpv[1][pixelType].size,
                                             .pCode = kSpv[1][pixelType].code };
        VK_CHECK(d->vk->vkCreateShaderModule(device, &smci, nullptr, &d->prescreenModule));
    }
    {
        const VkShaderModuleCreateInfo smci{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                             .codeSize = kSpv[2][pixelType].size,
                                             .pCode = kSpv[2][pixelType].code };
        VK_CHECK(d->vk->vkCreateShaderModule(device, &smci, nullptr, &d->predictModule));
    }

    VkDescriptorSetLayoutBinding bindings[BindCount];
    for (uint32_t i = 0; i < BindCount; i++)
        bindings[i] = VkDescriptorSetLayoutBinding{ .binding = i,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .descriptorCount = 1,
                                                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    const VkDescriptorSetLayoutCreateInfo dslci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                 .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                                 .bindingCount = BindCount,
                                                 .pBindings = bindings };
    VK_CHECK(d->vk->vkCreateDescriptorSetLayout(device, &dslci, nullptr, &d->dsl));

    const VkPushConstantRange pcr{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants) };
    const VkPipelineLayoutCreateInfo plci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = 1,
                                           .pSetLayouts = &d->dsl,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &pcr };
    VK_CHECK(d->vk->vkCreatePipelineLayout(device, &plci, nullptr, &d->pipelineLayout));

    static constexpr VkSpecializationMapEntry specEntries[] = {
        { 0, offsetof(SpecData, wgSize), sizeof(uint32_t) },
        { 1, offsetof(SpecData, pscrn), sizeof(int32_t) },
        { 2, offsetof(SpecData, xdim), sizeof(int32_t) },
        { 3, offsetof(SpecData, ydim), sizeof(int32_t) },
        { 4, offsetof(SpecData, nns), sizeof(int32_t) },
        { 5, offsetof(SpecData, qual), sizeof(int32_t) },
        { 6, offsetof(SpecData, sgSize), sizeof(int32_t) },
        { 7, offsetof(SpecData, subgroups), sizeof(int32_t) },
        { 8, offsetof(SpecData, useList), sizeof(VkBool32) },
    };
    constexpr uint32_t specEntryCount = static_cast<uint32_t>(std::size(specEntries));

    const uint32_t sgSize = d->subgroupSize;
    const uint32_t maxWG = std::min(maxWGInvocations, maxWGSizeX);
    d->prescreenWG = std::min(128u, maxWG);

    // Predict: PX pixels per subgroup (blocked GEMM).
    const uint32_t gemvPPL = (static_cast<uint32_t>(nns) + sgSize - 1) / sgSize;
    const uint32_t gemvFS = static_cast<uint32_t>(xdim * ydim);
    // Must match PX in predict.comp.
    const uint32_t gemvPX = (d->pixelType != 2 && gemvPPL <= 2 && gemvFS <= 128) ? 8 : 4;
    const uint32_t subgroupsPerWG = std::max(1u, std::min(4u, maxWG / sgSize));
    const uint32_t predictWG = sgSize * subgroupsPerWG;
    d->pixelsPerPredictWG = subgroupsPerWG * gemvPX;

    // Note: constant_id 7 differs per kernel — the prescreen kernel uses it
    // as "pixels per predict workgroup" (indirect dispatch sizing), the GEMV
    // predict kernel as its actual subgroup count.
    const SpecData spec{ .wgSize = 0, // filled in per pipeline
                         .pscrn = d->pscrn,
                         .xdim = xdim,
                         .ydim = ydim,
                         .nns = nns,
                         .qual = d->qual,
                         .sgSize = static_cast<int32_t>(sgSize),
                         .subgroups = 0, // filled in per pipeline
                         .useList = d->pscrn > 0 ? VK_TRUE : VK_FALSE };

    auto makePipeline = [&](VkShaderModule module, const SpecData& pspec, bool pinSubgroups, VkPipeline* out) {
        const VkSpecializationInfo specInfo{ .mapEntryCount = specEntryCount,
                                             .pMapEntries = specEntries,
                                             .dataSize = sizeof(pspec),
                                             .pData = &pspec };
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .requiredSubgroupSize = sgSize };
        const VkComputePipelineCreateInfo cpci{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .pNext = (pinSubgroups && d->canRequireSubgroupSize) ? &reqSg : nullptr,
                       .flags = pinSubgroups
                                    ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
                                    : 0,
                       .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                       .module = module,
                       .pName = "main",
                       .pSpecializationInfo = &specInfo },
            .layout = d->pipelineLayout,
        };
        VK_CHECK(d->vk->vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, out));
    };

    {
        SpecData pspec = spec; // pad/rowcopy have fixed local sizes; spec constants unused
        makePipeline(d->padModule, pspec, false, &d->padPipe);
        makePipeline(d->rowcopyModule, pspec, false, &d->rowcopyPipe);
    }
    if (d->pscrn > 0) {
        SpecData pspec = spec;
        pspec.wgSize = d->prescreenWG;
        pspec.subgroups = static_cast<int32_t>(d->pixelsPerPredictWG);
        makePipeline(d->prescreenModule, pspec, false, &d->prescreenPipe);
    }
    {
        // The GEMV kernel relies on subgroup-per-pixel mapping, so it pins
        // the subgroup size and requires full subgroups (both core features
        // the VapourSynth device always enables).
        SpecData pspec = spec;
        pspec.wgSize = predictWG;
        pspec.subgroups = static_cast<int32_t>(subgroupsPerWG);
        makePipeline(d->predictModule, pspec, true, &d->predictPipe);
    }

}

int getIntDef(const VSAPI* vsapi, const VSMap* in, const char* name, int def) {
    int err;
    int v = vsapi->mapGetIntSaturated(in, name, 0, &err);
    return err ? def : v;
}

void VS_CC nnedi3Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<NNEDI3Data>();
    int err;

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            (d->vi.format.sampleType == stInteger && d->vi.format.bitsPerSample > 16) ||
            (d->vi.format.sampleType == stFloat && d->vi.format.bitsPerSample != 16 && d->vi.format.bitsPerSample != 32))
            throw std::runtime_error("only constant format 8-16 bit integer and 16/32 bit float input supported");

        d->field = vsapi->mapGetIntSaturated(in, "field", 0, nullptr);

        d->dh = !!vsapi->mapGetInt(in, "dh", 0, &err);

        const int numPlanes = vsapi->mapNumElements(in, "planes");

        bool process[3] = {};
        for (int i = 0; i < 3; i++)
            process[i] = (numPlanes <= 0);

        for (int i = 0; i < numPlanes; i++) {
            const int plane = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);

            if (plane < 0 || plane >= d->vi.format.numPlanes)
                throw std::runtime_error("plane index out of range");

            if (process[plane])
                throw std::runtime_error("plane specified twice");

            process[plane] = true;
        }

        const int nsize = getIntDef(vsapi, in, "nsize", 6);
        const int nns = getIntDef(vsapi, in, "nns", 1);
        d->qual = getIntDef(vsapi, in, "qual", 1);
        const int etype = getIntDef(vsapi, in, "etype", 0);
        d->pscrn = getIntDef(vsapi, in, "pscrn", 2);
        const int numStreams = getIntDef(vsapi, in, "num_streams", 2);

        if (d->field < 0 || d->field > 3)
            throw std::runtime_error("field must be 0, 1, 2, or 3");

        if (!d->dh)
            for (int plane = 0; plane < d->vi.format.numPlanes; plane++)
                if (process[plane] && ((d->vi.height >> (plane > 0 ? d->vi.format.subSamplingH : 0)) & 1))
                    throw std::runtime_error("plane's height must be mod 2 when dh=False");

        if (d->dh && d->field > 1)
            throw std::runtime_error("field must be 0 or 1 when dh=True");

        if (nsize < 0 || nsize > 6)
            throw std::runtime_error("nsize must be between 0 and 6 (inclusive)");

        if (nns < 0 || nns > 4)
            throw std::runtime_error("nns must be between 0 and 4 (inclusive)");

        if (d->qual < 1 || d->qual > 2)
            throw std::runtime_error("qual must be 1 or 2");

        if (etype < 0 || etype > 1)
            throw std::runtime_error("etype must be 0 or 1");

        if (d->pscrn < 0 || d->pscrn > 4)
            throw std::runtime_error("pscrn must be between 0 and 4 (inclusive)");

        if (numStreams < 1)
            throw std::runtime_error("num_streams must be greater than or equal to 1");

        if (d->field > 1) {
            if (d->vi.numFrames > INT_MAX / 2)
                throw std::runtime_error("resulting clip is too long");
            d->vi.numFrames *= 2;

            vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);
        }

        if (d->dh)
            d->vi.height *= 2;

        // Load the nnedi3 weights that ship next to the plugin binary.
        PrescreenerOldCoefficients psOld;
        PrescreenerNewCoefficients psNew[3];
        PredictorModel model;
        {
            const char* pluginPath = vsapi->getPluginPath(vsapi->getPluginByID("com.holywu.nnedi3vk", core));
            if (!pluginPath)
                throw std::runtime_error("cannot determine plugin path");
            std::filesystem::path weightsPath(std::u8string(reinterpret_cast<const char8_t*>(pluginPath)));
            weightsPath = weightsPath.parent_path() / "nnedi3_weights.bin";

            std::ifstream file(weightsPath, std::ios::binary);
            std::vector<float> fileData(NNEDI3_WEIGHTS_SIZE / sizeof(float));
            if (!file.read(reinterpret_cast<char*>(fileData.data()), NNEDI3_WEIGHTS_SIZE) ||
                file.gcount() != static_cast<std::streamsize>(NNEDI3_WEIGHTS_SIZE))
                throw std::runtime_error("error reading weights from " + weightsPath.string());

            readNNEDI3Weights(fileData.data(), nsize, nns, etype, psOld, psNew, model);
        }

        const bool isFloat = d->vi.format.sampleType == stFloat;
        const double pixelHalf = isFloat ? 0.5 : static_cast<double>((1 << d->vi.format.bitsPerSample) - 1) / 2.0;

        if (d->pscrn == 1)
            subtractMean(psOld, pixelHalf);
        else if (d->pscrn >= 2)
            subtractMean(psNew[d->pscrn - 2], pixelHalf);
        subtractMean(model);

        if (d->vi.format.bytesPerSample == 1)
            d->pixelType = 0;
        else if (d->vi.format.bytesPerSample == 2 && !isFloat)
            d->pixelType = 1;
        else if (d->vi.format.bytesPerSample == 2)
            d->pixelType = 2;
        else
            d->pixelType = 3;

        d->peak = (1 << d->vi.format.bitsPerSample) - 1;

        d->profile = envFlag("NNEDI3VK_PROFILE");

        // The core's device, dispatch table and queue.
        char verr[512] = { 0 };
        d->vkapi = vsapi->getVulkanAPI(VSVULKAN_API_VERSION);
        if (!d->vkapi)
            throw std::runtime_error("Vulkan API not available in this core");
        if (d->vkapi->getVulkanHandles(core, &d->h, verr, sizeof(verr)))
            throw std::runtime_error(verr);
        d->vk = d->vkapi->getVulkanFunctions(core, verr, sizeof(verr));
        if (!d->vk)
            throw std::runtime_error(verr);

        const VkDeviceQueueInfo2 queueInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
                                            .queueFamilyIndex = d->h.computeQueueFamily,
                                            .queueIndex = d->h.computeQueueIndex };
        d->vk->vkGetDeviceQueue2(d->h.device, &queueInfo, &d->computeQueue);

        // Subgroup geometry and limits from the physical device. The
        // subgroup size control features are part of the core's required set,
        // so pinning is only limited by the stage support property.
        uint32_t maxWGInvocations, maxWGSizeX;
        {
            VkPhysicalDeviceVulkan13Properties props13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
            VkPhysicalDeviceVulkan11Properties props11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
                                                        .pNext = &props13 };
            VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                                .pNext = &props11 };
            d->vk->vkGetPhysicalDeviceProperties2(d->h.physicalDevice, &props2);

            constexpr VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT |
                VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
            if ((props11.subgroupSupportedOperations & needed) != needed)
                throw std::runtime_error("device does not support subgroup arithmetic/ballot operations");

            d->canRequireSubgroupSize = (props13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
            if (d->canRequireSubgroupSize)
                d->subgroupSize = std::clamp(32u, props13.minSubgroupSize, props13.maxSubgroupSize);
            else
                d->subgroupSize = props11.subgroupSize;

            d->timestampPeriod = props2.properties.limits.timestampPeriod;
            maxWGInvocations = props2.properties.limits.maxComputeWorkGroupInvocations;
            maxWGSizeX = props2.properties.limits.maxComputeWorkGroupSize[0];
        }

        if (d->pixelType == 2) {
            // shaderFloat16 is optional on the core device, enabled when the
            // hardware has it; availability is the best signal reachable here.
            VkPhysicalDeviceVulkan12Features f12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
            VkPhysicalDeviceFeatures2 f2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12 };
            d->vk->vkGetPhysicalDeviceFeatures2(d->h.physicalDevice, &f2);
            if (!f12.shaderFloat16)
                throw std::runtime_error("FP16 input requires shaderFloat16 device support");
        }

        // Plane layout / buffer sizes.
        const int bps = d->vi.format.bytesPerSample;
        VkDeviceSize padOff = 0, devOff = 0;
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            PlaneSetup& p = d->planes[plane];
            p.process = process[plane];
            p.width = d->vi.width >> (plane > 0 ? d->vi.format.subSamplingW : 0);
            p.height = d->vi.height >> (plane > 0 ? d->vi.format.subSamplingH : 0);
            if (!p.process)
                continue;

            p.rows = p.height / 2;
            p.padStride = (p.width + MARGIN_H * 2 + 15) & ~15;
            p.padHeight = p.rows + MARGIN_V * 2;

            p.padBytes = static_cast<VkDeviceSize>(p.padStride) * p.padHeight * bps;
            p.padOffset = suballoc(padOff, p.padBytes);

            if (d->pscrn > 0) {
                p.listBytes = static_cast<VkDeviceSize>(p.width) * p.rows * sizeof(uint32_t);
                p.listOffset = suballoc(devOff, p.listBytes);
                p.cntOffset = suballoc(devOff, 16);
            }
        }
        d->padSize = std::max(padOff, BUF_ALIGN);
        d->devbufSize = std::max(devOff, BUF_ALIGN);

        // Build the weight blobs.
        //
        // Prescreener: the flat float layout described in prescreen.comp,
        // with the layer-0 kernel transposed to [k][4] so the kernel fetches
        // all four neurons' weights for a window element in one vec4 load.
        std::vector<float> psBlob;
        if (d->pscrn == 1) {
            psBlob.resize(sizeof(psOld) / sizeof(float));
            const PrescreenerOldCoefficients& ps = psOld;
            float* p = psBlob.data();
            for (unsigned k = 0; k < 48; k++)
                for (unsigned n = 0; n < 4; n++)
                    *p++ = ps.kernel_l0[n][k];
            std::memcpy(p, ps.bias_l0, sizeof(psOld) - sizeof(ps.kernel_l0));
        } else if (d->pscrn >= 2) {
            psBlob.resize(sizeof(psNew[0]) / sizeof(float));
            const PrescreenerNewCoefficients& ps = psNew[d->pscrn - 2];
            float* p = psBlob.data();
            for (unsigned k = 0; k < 64; k++)
                for (unsigned n = 0; n < 4; n++)
                    *p++ = ps.kernel_l0[n][k];
            std::memcpy(p, ps.bias_l0, sizeof(psNew[0]) - sizeof(ps.kernel_l0));
        } else {
            psBlob.resize(1, 0.0f);
        }

        // Predictor: transposed (softmax, elliott) pairs plus interleaved
        // bias/rowsum pairs; see predict.comp for the exact layouts.
        const unsigned fs = model.xdim * model.ydim;
        const unsigned N = model.nns;
        const unsigned numQ = static_cast<unsigned>(d->qual);

        std::vector<float> pdBias(numQ * 4 * N);
        for (unsigned q = 0; q < numQ; q++) {
            const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
            const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
            const float* smB = q ? model.softmax_bias_q2.data() : model.softmax_bias_q1.data();
            const float* elB = q ? model.elliott_bias_q2.data() : model.elliott_bias_q1.data();

            for (unsigned p = 0; p < N; p++) {
                pdBias[(q * 2 * N + p) * 2 + 0] = smB[p];
                pdBias[(q * 2 * N + p) * 2 + 1] = elB[p];

                double smSum = 0.0, elSum = 0.0;
                for (unsigned k = 0; k < fs; k++) {
                    smSum += sm[p * fs + k];
                    elSum += el[p * fs + k];
                }
                pdBias[(q * 2 * N + N + p) * 2 + 0] = static_cast<float>(smSum);
                pdBias[(q * 2 * N + N + p) * 2 + 1] = static_cast<float>(elSum);
            }
        }

        std::vector<float> pdW32;
        std::vector<uint16_t> pdW16;
        if (d->pixelType == 2) {
            pdW16.resize(static_cast<size_t>(numQ) * (fs / 2) * N * 4);
            for (unsigned q = 0; q < numQ; q++) {
                const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
                const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
                for (unsigned k2 = 0; k2 < fs / 2; k2++)
                    for (unsigned p = 0; p < N; p++) {
                        const size_t base = ((static_cast<size_t>(q) * (fs / 2) + k2) * N + p) * 4;
                        pdW16[base + 0] = floatToHalf(sm[p * fs + k2 * 2 + 0]);
                        pdW16[base + 1] = floatToHalf(sm[p * fs + k2 * 2 + 1]);
                        pdW16[base + 2] = floatToHalf(el[p * fs + k2 * 2 + 0]);
                        pdW16[base + 3] = floatToHalf(el[p * fs + k2 * 2 + 1]);
                    }
            }
        } else {
            pdW32.resize(static_cast<size_t>(numQ) * fs * N * 2);
            for (unsigned q = 0; q < numQ; q++) {
                const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
                const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
                for (unsigned k = 0; k < fs; k++)
                    for (unsigned p = 0; p < N; p++) {
                        const size_t base = ((static_cast<size_t>(q) * fs + k) * N + p) * 2;
                        pdW32[base + 0] = sm[p * fs + k];
                        pdW32[base + 1] = el[p * fs + k];
                    }
            }
        }

        // Assemble the combined weights buffer.
        const void* pdWData;
        if (d->pixelType == 2) {
            d->pdWBytes = pdW16.size() * sizeof(uint16_t);
            pdWData = pdW16.data();
        } else {
            d->pdWBytes = pdW32.size() * sizeof(float);
            pdWData = pdW32.data();
        }
        d->psWBytes = psBlob.size() * sizeof(float);
        d->pdBBytes = pdBias.size() * sizeof(float);

        VkDeviceSize wOff = 0;
        d->psWOffset = suballoc(wOff, d->psWBytes);
        d->pdWOffset = suballoc(wOff, d->pdWBytes);
        d->pdBOffset = suballoc(wOff, d->pdBBytes);

        std::vector<uint8_t> weightsBlob(wOff);
        std::memcpy(weightsBlob.data() + d->psWOffset, psBlob.data(), d->psWBytes);
        std::memcpy(weightsBlob.data() + d->pdWOffset, pdWData, d->pdWBytes);
        std::memcpy(weightsBlob.data() + d->pdBOffset, pdBias.data(), d->pdBBytes);

        // The pool comes first: the weights upload records through it, and every
        // frame afterwards does too.
        d->execPool = d->vkapi->createGPUExecPool(core, vqCompute, numStreams, verr, sizeof(verr));
        if (!d->execPool)
            throw std::runtime_error("exec pool: "s + verr);

        uploadWeights(d.get(), core, weightsBlob.data(), weightsBlob.size());

        setupVulkanObjects(d.get(), static_cast<int32_t>(model.xdim),
                           static_cast<int32_t>(model.ydim), static_cast<int32_t>(model.nns),
                           maxWGInvocations, maxWGSizeX);
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, ("NNEDI3VK: "s + e.what()).c_str());
        nnedi3Free(d.release(), core, vsapi);
        return;
    }

    VSFilterDependency deps[] = { { d->node, d->field > 1 ? rpGeneral : rpStrictSpatial } };
    vsapi->createVideoFilterEx(out, "NNEDI3", &d->vi, nnedi3GetFrame, nnedi3Free, fmParallel, ffGPUOutput, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.nnedi3vk",
                         "nnedi3vk",
                         "Neural Network Edge Directed Interpolation 3, using Vulkan compute",
                         VS_MAKE_VERSION(2, 0),
                         VAPOURSYNTH_API_VERSION,
                         0,
                         plugin);

    vspapi->registerFunction("NNEDI3",
                             "clip:vnode:gpu;"
                             "field:int;"
                             "dh:int:opt;"
                             "planes:int[]:opt;"
                             "nsize:int:opt;"
                             "nns:int:opt;"
                             "qual:int:opt;"
                             "etype:int:opt;"
                             "pscrn:int:opt;"
                             "num_streams:int:opt;",
                             "clip:vnode:gpu;",
                             nnedi3Create,
                             nullptr,
                             plugin);
}
