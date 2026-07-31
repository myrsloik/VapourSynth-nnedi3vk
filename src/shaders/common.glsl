// Shared definitions for nnedi3vk compute shaders.
// PIXEL_TYPE: 0 = uint8, 1 = uint16, 2 = float16, 3 = float32

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require

#if PIXEL_TYPE == 2
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if PIXEL_TYPE == 0
#define pix_t uint8_t
#define INT_FMT 1
#elif PIXEL_TYPE == 1
#define pix_t uint16_t
#define INT_FMT 1
#elif PIXEL_TYPE == 2
#define pix_t float16_t
#define INT_FMT 0
#else
#define pix_t float
#define INT_FMT 0
#endif

// Must match MARGIN_H/MARGIN_V in nnedi3vk.cpp.
const int MARGIN_H = 24; // widest predictor window is 48 -> x-23 .. x+24
const int MARGIN_V = 3;  // tallest predictor window is 6 rows

// Specialization constants shared by all kernels (a kernel may leave some
// unused). IDs must match SpecData/specEntries in nnedi3vk.cpp; constant_id 0
// is each kernel's local_size_x.
layout(constant_id = 1) const int PSCRN = 2;     // 0 = none, 1 = old, >= 2 = new
layout(constant_id = 2) const int XDIM = 32;     // predictor window width
layout(constant_id = 3) const int YDIM = 4;      // predictor window height
layout(constant_id = 4) const int NNS = 32;      // neurons per activation type
layout(constant_id = 5) const int QUAL = 1;      // number of quality passes
layout(constant_id = 6) const int SGSIZE = 32;   // required subgroup size (predict kernel)
layout(constant_id = 7) const int SUBGROUPS = 4; // predict: subgroups (= pixels) per workgroup

// Field order and types must match struct PushConstants in nnedi3vk.cpp
// (scalar layout makes the correspondence purely positional).
layout(push_constant, scalar) uniform PC {
    int width;     // output plane width
    int rows;      // interpolated rows (= field height)
    int padStride; // padded plane stride in elements
    int peak;      // integer clamp ceiling (255 / 65535)
    int srcStride; // pad/rowcopy: source plane stride in elements
    int rowOff;    // pad/rowcopy: field row f maps to source row rowOff + rowScale * f
    int rowScale;
    int fp;        // pad kernel only: znedi3 PadFilter parity (= !field parity)
    int dstBase;   // output element index of interpolated (or copied) row 0
    int dstPitch;  // output elements between consecutive interpolated rows
} pc;

// Padded source plane: (rows + 2*MARGIN_V) x padStride in the native pixel
// type. The CPU builds it so that the interpolated output row r sits between
// padded rows r + MARGIN_V - 1 and r + MARGIN_V (same geometry as znedi3:
// cubic taps rows r+1 .. r+4, the ydim=6 predictor window is rows r .. r+5).
layout(binding = 0, scalar) readonly restrict buffer PadBuf { pix_t padBuf[]; };

// Interpolated rows only, width x rows, native pixel type.
layout(binding = 1, scalar) writeonly restrict buffer DstBuf { pix_t dstBuf[]; };

float loadPad(int row, int col) {
    return float(padBuf[row * pc.padStride + col]);
}

pix_t storePix(float v) {
#if INT_FMT
    // Mirrors the reference float_to_integer: clamp to the storage maximum,
    // then round to nearest even (lrint).
    return pix_t(uint(roundEven(clamp(v, 0.0, float(pc.peak)))));
#else
    return pix_t(v);
#endif
}

// The output binding is the destination plane itself; interpolated pixel
// idx = r * width + x lands at output row dstBase + r * dstPitch, so the
// kernels write their strided half of the frame directly with no packed
// intermediate or copy pass.
void storeDst(uint idx, float v) {
    const int r = int(idx) / pc.width;
    const int x = int(idx) % pc.width;
    dstBuf[pc.dstBase + r * pc.dstPitch + x] = storePix(v);
}

float elliott(float x) {
    return x / (1.0 + abs(x));
}

// Top-left corner (col0, row0) of pixel idx's predictor window in the padded
// plane; matches znedi3's tap layout (ydim=6 windows start one row higher).
ivec2 windowOrigin(uint idx) {
    const int r = int(idx) / pc.width;
    const int x = int(idx) % pc.width;
    return ivec2(x - (XDIM / 2 - 1) + MARGIN_H, r + (YDIM == 6 ? 0 : 1));
}

// Window mean/std, matching the reference: mstd2 is 1/std, or std/1/inv-std all
// zero when the variance underflows FLT_EPSILON.
void computeMstd(float mean, float variance, out float mstd0, out float mstd1, out float mstd2) {
    mstd0 = mean;
    if (variance < 1.1920929e-7) { // FLT_EPSILON, as in the reference
        mstd1 = 0.0;
        mstd2 = 0.0;
    } else {
        mstd1 = sqrt(variance);
        mstd2 = 1.0 / mstd1;
    }
}

// One neuron's contribution to the wae5 softmax reduction.
void wae5Accum(float actS, float actE, inout float vsum, inout float wsum) {
    const float e = exp(clamp(actS, -80.0, 80.0));
    vsum = fma(e, elliott(actE), vsum);
    wsum += e;
}

// Final wae5 blend into the output value (per quality pass).
float wae5Blend(float vsum, float wsum, float mstd0, float mstd1) {
    return wsum > 1e-10 ? fma((5.0 * vsum) / wsum, mstd1, mstd0) : mstd0;
}
