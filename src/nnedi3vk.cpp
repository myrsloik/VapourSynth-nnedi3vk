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

// Implemented on the core's gpufilter.h declaration driver, verbatim -- no
// local extensions: the filter declares its programs, passes and operands
// once, plus callbacks for push constants, per-frame field parity and frame
// property fixup, and the driver owns everything the previous version
// recorded by hand -- output allocation with plane sharing, the exec pool and
// its contexts, source retention, scratch and constant buffers, barriers,
// descriptor pushes, dispatch and submission.
//
// The predict kernel is dispatched over the worst case (every interpolated
// pixel) instead of the indirect dispatch earlier versions used: it already
// bounds itself by the prescreener's compacted count, so the workgroups past
// the list retire on one uniform load -- tens of microseconds a frame at
// 1080p, which is what expressing the filter without the indirect dispatch
// extension costs. The prescreener accordingly no longer maintains dispatch
// arguments, only the append counter.
//
// What the declaration cannot say, callbacks say: the pad plane and the pixel
// list are not shaped like any frame plane (Pass::reshape), and the list
// counter is reset by a one-thread kernel the driver compiles from source at
// creation instead of a vkCmdFillBuffer. dh with unprocessed planes zero
// fills them through a dedicated pass: the driver shares unprocessed planes
// from the source, which a doubled height makes impossible, and the
// hand-recorded version's "left unwritten" is not part of its model. The GPU
// profiling of the hand-recorded version (NNEDI3VK_PROFILE) is gone: the
// driver has no timestamp hook, and it was a debug aid.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define VS_USE_API_43
#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

#include "gpufilter.h"

using namespace std::string_literals;

// Must match common.glsl.
constexpr int MARGIN_H = 24;
constexpr int MARGIN_V = 3;

#include "shadersources.h"

// One kernel's source, ready for the core's compiler: the version directive the generator
// stripped, then the pixel type the kernel specialises on, then the body. The define has to sit
// here rather than in the file because GLSL requires #version first, and the core's compiler
// takes a single string with no include handler -- see embed_shaders.py, which resolved the
// common.glsl include at build time for the same reason.
std::string shaderSource(const char *body, int pixelType) {
    return "#version 460\n#define PIXEL_TYPE " + std::to_string(pixelType) + "\n" + body;
}

// Indexed [kernel][pixelType]; kernels: 0 = pad, 1 = prescreen, 2 = predict. The sources are
// pixel type agnostic, the define is what specialises them, so one row is one kernel repeated.
constexpr const char *kKernelSrc[3] = { kPadSrc, kPrescreenSrc, kPredictSrc };

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
// Push constants / specialization

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

constexpr VkSpecializationMapEntry specEntries[] = {
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

// Resets the prescreener's append counter; replaces the vkCmdFillBuffer of
// the hand-recorded version, since a declared pass is all the driver offers --
// and one thread of one kernel is exactly as cheap. Words 1..3 of the block
// are dead space (the retired indirect dispatch arguments), cleared along
// with the counter to keep the kernels' 16 byte block layout unchanged.
const char resetGlsl[] =
    "#version 460\n"
    "layout(local_size_x = 1) in;\n"
    "layout(std430, binding = 0) writeonly buffer Cnt { uint cnt[]; };\n"
    "void main() { cnt[0] = 0u; cnt[1] = 0u; cnt[2] = 1u; cnt[3] = 1u; }\n";

// Zero fills a plane as 4 byte words (strides are at least 32 byte aligned,
// so every plane's bytes divide); pc.width carries the word count. Only used
// for dh with unprocessed planes, whose doubled height makes sharing the
// source plane impossible.
const char zeroGlsl[] =
    "#version 460\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) writeonly buffer Dst { uint d[]; };\n"
    "layout(push_constant, std430) uniform PC { int width; } pc;\n"
    "void main() {\n"
    "    if (gl_GlobalInvocationID.x < uint(pc.width))\n"
    "        d[gl_GlobalInvocationID.x] = 0u;\n"
    "}\n";

//////////////////////////////////////////
// The declaration's shared state

// Everything the desc callbacks need per dispatch, computed once at creation
// and shared by the capturing lambdas (the driver copies the desc, so the
// lambdas and this state live exactly as long as the filter instance).
struct NNEDI3State {
    bool process[3] = {};
    int width[3] = {}, rows[3] = {};
    int padStride[3] = {}, padHeight[3] = {};
    int outHeight[3] = {}; // full output plane height, set for every plane
    int bps = 0, peak = 0;
    int field = 0, pscrn = 0;
    bool dh = false;
    // Pass indices in the declared pass list, -1 when absent.
    int passRowcopy = -1, passPad = -1, passPrescreen = -1, passPredict = -1, passZero = -1;
};

//////////////////////////////////////////
// Creation

int getIntDef(const VSAPI* vsapi, const VSMap* in, const char* name, int def) {
    int err;
    int v = vsapi->mapGetIntSaturated(in, name, 0, &err);
    return err ? def : v;
}

void VS_CC nnedi3Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    VSNode* node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    int err;

    try {
        VSVideoInfo vi = *vsapi->getVideoInfo(node);

        if (!vsh::isConstantVideoFormat(&vi) ||
            (vi.format.sampleType == stInteger && vi.format.bitsPerSample > 16) ||
            (vi.format.sampleType == stFloat && vi.format.bitsPerSample != 16 && vi.format.bitsPerSample != 32))
            throw std::runtime_error("only constant format 8-16 bit integer and 16/32 bit float input supported");

        const int field = vsapi->mapGetIntSaturated(in, "field", 0, nullptr);
        const bool dh = !!vsapi->mapGetInt(in, "dh", 0, &err);

        const int numPlanes = vsapi->mapNumElements(in, "planes");

        bool process[3] = {};
        for (int i = 0; i < 3; i++)
            process[i] = (numPlanes <= 0);

        for (int i = 0; i < numPlanes; i++) {
            const int plane = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);

            if (plane < 0 || plane >= vi.format.numPlanes)
                throw std::runtime_error("plane index out of range");

            if (process[plane])
                throw std::runtime_error("plane specified twice");

            process[plane] = true;
        }

        const int nsize = getIntDef(vsapi, in, "nsize", 6);
        const int nns = getIntDef(vsapi, in, "nns", 1);
        const int qual = getIntDef(vsapi, in, "qual", 1);
        const int etype = getIntDef(vsapi, in, "etype", 0);
        const int pscrn = getIntDef(vsapi, in, "pscrn", 2);

        if (field < 0 || field > 3)
            throw std::runtime_error("field must be 0, 1, 2, or 3");

        if (!dh)
            for (int plane = 0; plane < vi.format.numPlanes; plane++)
                if (process[plane] && ((vi.height >> (plane > 0 ? vi.format.subSamplingH : 0)) & 1))
                    throw std::runtime_error("plane's height must be mod 2 when dh=False");

        if (dh && field > 1)
            throw std::runtime_error("field must be 0 or 1 when dh=True");

        if (nsize < 0 || nsize > 6)
            throw std::runtime_error("nsize must be between 0 and 6 (inclusive)");

        if (nns < 0 || nns > 4)
            throw std::runtime_error("nns must be between 0 and 4 (inclusive)");

        if (qual < 1 || qual > 2)
            throw std::runtime_error("qual must be 1 or 2");

        if (etype < 0 || etype > 1)
            throw std::runtime_error("etype must be 0 or 1");

        if (pscrn < 0 || pscrn > 4)
            throw std::runtime_error("pscrn must be between 0 and 4 (inclusive)");

        if (field > 1) {
            if (vi.numFrames > INT_MAX / 2)
                throw std::runtime_error("resulting clip is too long");
            vi.numFrames *= 2;

            vsh::muldivRational(&vi.fpsNum, &vi.fpsDen, 2, 1);
        }

        if (dh)
            vi.height *= 2;

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

        const bool isFloat = vi.format.sampleType == stFloat;
        const double pixelHalf = isFloat ? 0.5 : static_cast<double>((1 << vi.format.bitsPerSample) - 1) / 2.0;

        if (pscrn == 1)
            subtractMean(psOld, pixelHalf);
        else if (pscrn >= 2)
            subtractMean(psNew[pscrn - 2], pixelHalf);
        subtractMean(model);

        int pixelType;
        if (vi.format.bytesPerSample == 1)
            pixelType = 0;
        else if (vi.format.bytesPerSample == 2 && !isFloat)
            pixelType = 1;
        else if (vi.format.bytesPerSample == 2)
            pixelType = 2;
        else
            pixelType = 3;

        // Device geometry the specialization needs: subgroup size and control,
        // workgroup limits, and the optional shaderFloat16. All reachable
        // through the public handle/dispatch queries; the driver needs none of
        // it, only the resulting Program fields.
        char verr[512] = { 0 };
        const VSVULKANAPI* vkapi = vsapi->getVulkanAPI();
        if (!vkapi)
            throw std::runtime_error("Vulkan API not available in this core");
        VSVulkanCoreHandles h{};
        if (vkapi->getVulkanHandles(core, &h, verr, sizeof(verr)))
            throw std::runtime_error(verr);
        const VSVulkanFunctions* vk = vkapi->getVulkanFunctions(core, verr, sizeof(verr));
        if (!vk)
            throw std::runtime_error(verr);

        uint32_t subgroupSize, maxWGInvocations, maxWGSizeX;
        bool canRequireSubgroupSize;
        {
            VkPhysicalDeviceVulkan13Properties props13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
            VkPhysicalDeviceVulkan11Properties props11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
                                                        .pNext = &props13 };
            VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                                .pNext = &props11 };
            vk->vkGetPhysicalDeviceProperties2(h.physicalDevice, &props2);

            constexpr VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT |
                VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
            if ((props11.subgroupSupportedOperations & needed) != needed)
                throw std::runtime_error("device does not support subgroup arithmetic/ballot operations");

            canRequireSubgroupSize = (props13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
            if (canRequireSubgroupSize)
                subgroupSize = std::clamp(32u, props13.minSubgroupSize, props13.maxSubgroupSize);
            else
                subgroupSize = props11.subgroupSize;

            maxWGInvocations = props2.properties.limits.maxComputeWorkGroupInvocations;
            maxWGSizeX = props2.properties.limits.maxComputeWorkGroupSize[0];
        }

        if (pixelType == 2) {
            // shaderFloat16 is optional on the core device, enabled when the
            // hardware has it; availability is the best signal reachable here.
            VkPhysicalDeviceVulkan12Features f12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
            VkPhysicalDeviceFeatures2 f2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12 };
            vk->vkGetPhysicalDeviceFeatures2(h.physicalDevice, &f2);
            if (!f12.shaderFloat16)
                throw std::runtime_error("FP16 input requires shaderFloat16 device support");
        }

        // Plane geometry and the derived scratch sizes.
        auto st = std::make_shared<NNEDI3State>();
        st->bps = vi.format.bytesPerSample;
        st->peak = isFloat ? 0 : (1 << vi.format.bitsPerSample) - 1;
        st->field = field;
        st->pscrn = pscrn;
        st->dh = dh;

        VkDeviceSize maxPadBytes = 0, maxListBytes = 0;
        for (int plane = 0; plane < vi.format.numPlanes; plane++) {
            st->process[plane] = process[plane];
            st->width[plane] = vi.width >> (plane > 0 ? vi.format.subSamplingW : 0);
            const int height = vi.height >> (plane > 0 ? vi.format.subSamplingH : 0);
            st->outHeight[plane] = height;
            if (!process[plane])
                continue;

            st->rows[plane] = height / 2;
            st->padStride[plane] = (st->width[plane] + MARGIN_H * 2 + 15) & ~15;
            st->padHeight[plane] = st->rows[plane] + MARGIN_V * 2;

            maxPadBytes = std::max(maxPadBytes,
                static_cast<VkDeviceSize>(st->padStride[plane]) * st->padHeight[plane] * st->bps);
            maxListBytes = std::max(maxListBytes,
                static_cast<VkDeviceSize>(st->width[plane]) * st->rows[plane] * sizeof(uint32_t));
        }

        // Build the weight blobs.
        //
        // Prescreener: the flat float layout described in prescreen.comp,
        // with the layer-0 kernel transposed to [k][4] so the kernel fetches
        // all four neurons' weights for a window element in one vec4 load.
        std::vector<float> psBlob;
        if (pscrn == 1) {
            psBlob.resize(sizeof(psOld) / sizeof(float));
            const PrescreenerOldCoefficients& ps = psOld;
            float* p = psBlob.data();
            for (unsigned k = 0; k < 48; k++)
                for (unsigned n = 0; n < 4; n++)
                    *p++ = ps.kernel_l0[n][k];
            std::memcpy(p, ps.bias_l0, sizeof(psOld) - sizeof(ps.kernel_l0));
        } else if (pscrn >= 2) {
            psBlob.resize(sizeof(psNew[0]) / sizeof(float));
            const PrescreenerNewCoefficients& ps = psNew[pscrn - 2];
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
        const unsigned numQ = static_cast<unsigned>(qual);

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

        if (pixelType == 2) {
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

        // Workgroup geometry, identical to the hand-recorded version.
        const uint32_t sgSize = subgroupSize;
        const uint32_t maxWG = std::min(maxWGInvocations, maxWGSizeX);
        const uint32_t prescreenWG = std::min(128u, maxWG);

        // Predict: PX pixels per subgroup (blocked GEMM).
        const uint32_t gemvPPL = (static_cast<uint32_t>(model.nns) + sgSize - 1) / sgSize;
        const uint32_t gemvFS = fs;
        // Must match PX in predict.comp.
        const uint32_t gemvPX = (pixelType != 2 && gemvPPL <= 2 && gemvFS <= 128) ? 8 : 4;
        const uint32_t subgroupsPerWG = std::max(1u, std::min(4u, maxWG / sgSize));
        const uint32_t predictWG = sgSize * subgroupsPerWG;
        const uint32_t pixelsPerPredictWG = subgroupsPerWG * gemvPX;

        // Note: constant_id 7 is the GEMV predict kernel's subgroups per
        // workgroup. The prescreen kernel declares it but no longer uses it
        // (it fed the retired indirect dispatch sizing); it still receives
        // "pixels per predict workgroup" for compatibility with the shared
        // spec table.
        const SpecData baseSpec{ .wgSize = 0,
                                 .pscrn = pscrn,
                                 .xdim = static_cast<int32_t>(model.xdim),
                                 .ydim = static_cast<int32_t>(model.ydim),
                                 .nns = static_cast<int32_t>(model.nns),
                                 .qual = qual,
                                 .sgSize = static_cast<int32_t>(sgSize),
                                 .subgroups = 0,
                                 .useList = pscrn > 0 ? VK_TRUE : VK_FALSE };
        auto specialize = [&](vsgpu::Program& prog, uint32_t wgSize, int32_t subgroups) {
            SpecData sd = baseSpec;
            sd.wgSize = wgSize;
            sd.subgroups = subgroups;
            prog.specData.resize(sizeof(sd));
            std::memcpy(prog.specData.data(), &sd, sizeof(sd));
            prog.specEntries.assign(std::begin(specEntries), std::end(specEntries));
        };

        //////////////////////////////////////////
        // The declaration itself.

        vsgpu::FilterDesc desc;
        desc.vi = vi;
        desc.nodes.push_back(node);
        // dh doubles the height, so an unprocessed plane cannot share the
        // source plane the way the driver models unprocessed planes; it is
        // produced too, by the zero fill pass below (the hand-recorded
        // version left it unwritten, which was documented as undefined).
        bool zeroPlanes = false;
        for (int p = 0; p < 3; p++) {
            desc.process[p] = process[p] || (dh && p < vi.format.numPlanes);
            zeroPlanes = zeroPlanes || (desc.process[p] && !process[p]);
        }

        // Scratch: the padded field plane (wider than any output plane, so
        // explicitly sized), the rejected-pixel list, and the 16 byte
        // append-counter block.
        desc.scratchCount = 3;
        desc.scratchDefs = {
            { maxPadBytes, 0 },
            { pscrn > 0 ? maxListBytes : 16, 0 },
            { 16, 0 },
        };
        constexpr int scratchPad = 0, scratchList = 1, scratchCnt = 2;

        // Constants: prescreener weights, predictor weights, predictor
        // bias/rowsums -- uploaded once by the driver, device local.
        auto toBytes = [](const void* data, size_t bytes) {
            std::vector<uint8_t> v(bytes);
            std::memcpy(v.data(), data, bytes);
            return v;
        };
        desc.constants.push_back(toBytes(psBlob.data(), psBlob.size() * sizeof(float)));
        if (pixelType == 2)
            desc.constants.push_back(toBytes(pdW16.data(), pdW16.size() * sizeof(uint16_t)));
        else
            desc.constants.push_back(toBytes(pdW32.data(), pdW32.size() * sizeof(float)));
        desc.constants.push_back(toBytes(pdBias.data(), pdBias.size() * sizeof(float)));
        constexpr int constPsW = 0, constPdW = 1, constPdB = 2;

        // Programs. localSizeX/Y are the dispatch divisors: for the network
        // kernels that is pixels per workgroup, not the specialized thread
        // count.
        vsgpu::Program rowcopyProg;
        // Byte addressed, so the pixel type it is specialised with is immaterial.
        rowcopyProg.glsl = shaderSource(kRowcopySrc, 0);
        rowcopyProg.storageBufferCount = 2;
        rowcopyProg.pushConstantBytes = sizeof(PushConstants);
        rowcopyProg.localSizeX = 64;
        rowcopyProg.localSizeY = 4;

        vsgpu::Program padProg;
        padProg.glsl = shaderSource(kKernelSrc[0], pixelType);
        padProg.storageBufferCount = 2;
        padProg.pushConstantBytes = sizeof(PushConstants);
        padProg.localSizeX = 16;
        padProg.localSizeY = 16;

        vsgpu::Program predictProg;
        predictProg.glsl = shaderSource(kKernelSrc[2], pixelType);
        predictProg.storageBufferCount = 7;
        predictProg.pushConstantBytes = sizeof(PushConstants);
        predictProg.localSizeX = pixelsPerPredictWG;
        predictProg.localSizeY = 1;
        specialize(predictProg, predictWG, static_cast<int32_t>(subgroupsPerWG));
        // The GEMV kernel relies on subgroup-per-pixel mapping, so it pins
        // the subgroup size and requires full subgroups.
        predictProg.requiredSubgroupSize = canRequireSubgroupSize ? sgSize : 0;
        predictProg.requireFullSubgroups = true;

        desc.programs.push_back(rowcopyProg);
        desc.programs.push_back(padProg);
        const int progRowcopy = 0, progPad = 1;
        int progReset = -1, progPrescreen = -1, progZero = -1;
        if (zeroPlanes) {
            vsgpu::Program zeroProg;
            zeroProg.glsl = zeroGlsl;
            zeroProg.storageBufferCount = 1;
            zeroProg.pushConstantBytes = sizeof(PushConstants);
            zeroProg.localSizeX = 256;
            zeroProg.localSizeY = 1;
            desc.programs.push_back(zeroProg);
            progZero = static_cast<int>(desc.programs.size()) - 1;
        }
        if (pscrn > 0) {
            vsgpu::Program resetProg;
            resetProg.glsl = resetGlsl;
            resetProg.storageBufferCount = 1;
            resetProg.pushConstantBytes = 0;
            resetProg.localSizeX = 1;
            resetProg.localSizeY = 1;

            vsgpu::Program prescreenProg;
            prescreenProg.glsl = shaderSource(kKernelSrc[1], pixelType);
            prescreenProg.storageBufferCount = 7;
            prescreenProg.pushConstantBytes = sizeof(PushConstants);
            prescreenProg.localSizeX = prescreenWG;
            prescreenProg.localSizeY = 1;
            specialize(prescreenProg, prescreenWG, static_cast<int32_t>(pixelsPerPredictWG));

            desc.programs.push_back(resetProg);
            progReset = static_cast<int>(desc.programs.size()) - 1;
            desc.programs.push_back(prescreenProg);
            progPrescreen = static_cast<int>(desc.programs.size()) - 1;
        }
        desc.programs.push_back(predictProg);
        const int progPredict = static_cast<int>(desc.programs.size()) - 1;

        // Passes, in per-plane recording order. The network kernels bind the
        // full seven-operand table of the shaders' shared layout; the entries
        // a kernel does not declare are simply never read.
        const std::vector<vsgpu::Operand> kernelBindings = {
            vsgpu::Operand::scratch(scratchPad),
            vsgpu::Operand::output(),
            vsgpu::Operand::constant(constPsW),
            vsgpu::Operand::constant(constPdW),
            vsgpu::Operand::constant(constPdB),
            vsgpu::Operand::scratch(scratchList),
            vsgpu::Operand::scratch(scratchCnt),
        };

        // The interpolation passes run only for the planes the user named;
        // dh's zero filled planes are covered by their own pass instead.
        auto gatePlanes = [&process](vsgpu::Pass& pass) {
            for (int p = 0; p < 3; p++)
                pass.planes[p] = process[p];
        };

        if (zeroPlanes) {
            vsgpu::Pass pass;
            pass.program = progZero;
            pass.bindings = { vsgpu::Operand::output() };
            // Writes planes nothing else touches, so it needs no ordering.
            pass.independent = true;
            for (int p = 0; p < 3; p++)
                pass.planes[p] = desc.process[p] && !process[p];
            pass.reshape = [st](vsgpu::PassInfo& info) {
                info.width = info.strideElements[0] * static_cast<uint32_t>(st->bps) *
                    static_cast<uint32_t>(st->outHeight[info.plane]) / 4u;
                info.height = 1;
            };
            st->passZero = static_cast<int>(desc.passes.size());
            desc.passes.push_back(std::move(pass));
        }
        {
            vsgpu::Pass pass;
            pass.program = progRowcopy;
            pass.bindings = { vsgpu::Operand::source(), vsgpu::Operand::output() };
            // Reads only the source plane and writes only the kept rows, which
            // no other pass touches, so it needs no ordering against anything.
            pass.independent = true;
            gatePlanes(pass);
            pass.reshape = [st](vsgpu::PassInfo& info) {
                info.width = static_cast<uint32_t>(st->width[info.plane] * st->bps);
                info.height = static_cast<uint32_t>(st->rows[info.plane]);
            };
            st->passRowcopy = static_cast<int>(desc.passes.size());
            desc.passes.push_back(std::move(pass));
        }
        {
            vsgpu::Pass pass;
            pass.program = progPad;
            pass.bindings = { vsgpu::Operand::source(), vsgpu::Operand::scratch(scratchPad) };
            gatePlanes(pass);
            pass.reshape = [st](vsgpu::PassInfo& info) {
                info.width = static_cast<uint32_t>(st->padStride[info.plane]);
                info.height = static_cast<uint32_t>(st->padHeight[info.plane]);
            };
            st->passPad = static_cast<int>(desc.passes.size());
            desc.passes.push_back(std::move(pass));
        }
        if (pscrn > 0) {
            {
                vsgpu::Pass pass;
                pass.program = progReset;
                pass.bindings = { vsgpu::Operand::scratch(scratchCnt) };
                gatePlanes(pass);
                pass.reshape = [](vsgpu::PassInfo& info) { info.width = info.height = 1; };
                desc.passes.push_back(std::move(pass));
            }
            {
                vsgpu::Pass pass;
                pass.program = progPrescreen;
                pass.bindings = kernelBindings;
                gatePlanes(pass);
                pass.reshape = [st](vsgpu::PassInfo& info) {
                    const int pixPerThread = (st->pscrn == 1) ? 1 : 4;
                    info.width = static_cast<uint32_t>(st->rows[info.plane]) *
                        ((static_cast<uint32_t>(st->width[info.plane]) + pixPerThread - 1) / pixPerThread);
                    info.height = 1;
                };
                st->passPrescreen = static_cast<int>(desc.passes.size());
                desc.passes.push_back(std::move(pass));
            }
        }
        {
            vsgpu::Pass pass;
            pass.program = progPredict;
            pass.bindings = kernelBindings;
            gatePlanes(pass);
            // Dispatched over the worst case, every interpolated pixel. With a
            // prescreener the kernel bounds itself by the compacted count
            // (predCount), so the workgroups past the list retire on one
            // uniform load; without one the same expression IS the exact
            // grid. Earlier versions read the grid from the counter block via
            // vkCmdDispatchIndirect instead -- ceil(count / pixels per
            // workgroup) -- with the same kernel-side guard doing the real
            // bounding either way.
            pass.reshape = [st](vsgpu::PassInfo& info) {
                info.width = static_cast<uint32_t>(st->width[info.plane]) *
                    static_cast<uint32_t>(st->rows[info.plane]);
                info.height = 1;
            };
            st->passPredict = static_cast<int>(desc.passes.size());
            desc.passes.push_back(std::move(pass));
        }

        // Source field parity, mirroring vsznedi3's get_src_parity exactly.
        // parity == 1 means the source is (treated as) the bottom field.
        desc.frameParamCount = 1;
        desc.prepareFrame = [st](int n, const VSFrame* const* sources, [[maybe_unused]] int numSources,
                                 const VSAPI* vsapi, uint32_t* params, [[maybe_unused]] std::string& error) {
            const VSMap* srcProps = vsapi->getFramePropertiesRO(sources[0]);
            const int defaultParity = (st->field == 0 || st->field == 2) ? 1 : 0;
            int parity;
            int perr;
            if (st->dh) {
                const int fieldProp = vsapi->mapGetIntSaturated(srcProps, "_Field", 0, &perr);
                parity = perr ? defaultParity : fieldProp;
            } else if (st->field > 1) {
                const int fieldBased = vsapi->mapGetIntSaturated(srcProps, "_FieldBased", 0, &perr);
                parity = fieldBased == VSC_FIELD_BOTTOM ? 1 : fieldBased == VSC_FIELD_TOP ? 0 : defaultParity;
                if (n % 2)
                    parity = !parity;
            } else {
                parity = st->field == 0 ? 1 : 0;
            }
            params[0] = static_cast<uint32_t>(!!parity);
            return true;
        };

        // field > 1 doubles the output frame rate.
        if (field > 1)
            desc.mapFrame = [](int n, [[maybe_unused]] int clip, int frameOffset) { return n / 2 + frameOffset; };

        desc.fillPush = [st](const vsgpu::PassInfo& info, void* push) {
            const int p = info.plane;
            const int parity = static_cast<int>(info.frameParams[0]);
            PushConstants pc{};
            if (info.pass == st->passZero) {
                // Word count of the whole plane; matches the reshape above.
                pc.width = static_cast<int32_t>(info.strideElements[0]) * st->bps * st->outHeight[p] / 4;
            } else if (info.pass == st->passRowcopy) {
                // Byte addressed: every field of this kernel's push block is
                // in bytes, and source row r of the kept field maps to output
                // row parity + 2r (or 1:1 when dh).
                const int srcStrideB = static_cast<int>(info.strideElements[0]) * st->bps;
                const int dstStrideB = static_cast<int>(info.strideElements[1]) * st->bps;
                pc.width = st->width[p] * st->bps;
                pc.rows = st->rows[p];
                pc.rowOff = st->dh ? 0 : parity * srcStrideB;
                pc.rowScale = srcStrideB * (st->dh ? 1 : 2);
                pc.dstBase = parity * dstStrideB;
                pc.dstPitch = dstStrideB * 2;
            } else if (info.pass == st->passPad) {
                pc.width = st->width[p];
                pc.rows = st->rows[p];
                pc.padStride = st->padStride[p];
                pc.srcStride = static_cast<int>(info.strideElements[0]);
                pc.rowOff = st->dh ? 0 : parity;
                pc.rowScale = st->dh ? 1 : 2;
                pc.fp = !parity; // znedi3 PadFilter parity
            } else {
                // prescreen and predict: interpolated rows go straight into
                // the destination plane, strided; row r of the field lands at
                // output row !parity + 2r.
                const int dstStrideElems = static_cast<int>(info.strideElements[1]);
                pc.width = st->width[p];
                pc.rows = st->rows[p];
                pc.padStride = st->padStride[p];
                pc.peak = st->peak;
                pc.dstBase = !parity * dstStrideElems;
                pc.dstPitch = 2 * dstStrideElems;
            }
            std::memcpy(push, &pc, sizeof(pc));
        };

        desc.finishFrame = [st](int, VSFrame* dst, const VSFrame* const*, int, const uint32_t*,
                                [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
            VSMap* props = vsapi->getFramePropertiesRW(dst);
            vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
            vsapi->mapDeleteKey(props, "_Field");

            if (st->field > 1) {
                int errNum, errDen;
                int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
                int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
                if (!errNum && !errDen) {
                    vsh::muldivRational(&durationNum, &durationDen, 1, 2);
                    vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
                    vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
                }
            }
        };

        const VSFilterDependency deps[] = { { node, field > 1 ? rpGeneral : rpStrictSpatial } };
        std::string errorMessage;
        VSNode* created = vsgpu::createFilter("NNEDI3", desc, deps, 1, core, vsapi, errorMessage);
        if (!created) {
            // createFilter consumed the nodes on failure too.
            vsapi->mapSetError(out, ("NNEDI3VK: " + errorMessage).c_str());
            return;
        }
        vsapi->mapConsumeNode(out, "clip", created, maAppend);
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, ("NNEDI3VK: "s + e.what()).c_str());
        vsapi->freeNode(node);
        return;
    }
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
                             "pscrn:int:opt;",
                             "clip:vnode:gpu;",
                             nnedi3Create,
                             nullptr,
                             plugin);
}
