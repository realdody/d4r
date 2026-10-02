#include "hip_api.h"
#include "cuda_api.h"
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

// Public synthetic matrix fragments, independent of NVIDIA DLLs and weights.
static constexpr char source[] = R"PTX(
.version 7.0
.target sm_75
.address_size 64
.visible .entry d4r_f16_mma_reference(
 .param .u64 A, .param .u64 B, .param .u64 C, .param .u64 D)
{
 .reg .b32 %r<24>;
 .reg .b64 %rd<12>;
 .reg .b16 %h<4>;
 ld.param.u64 %rd0, [A];
 ld.param.u64 %rd1, [B];
 ld.param.u64 %rd2, [C];
 ld.param.u64 %rd3, [D];
 mov.u32 %r0, %ctaid.x;
 mov.u32 %r1, %tid.x;
 shr.u32 %r2, %r1, 2;
 and.b32 %r3, %r1, 3;
 shl.b32 %r3, %r3, 1;
 mul.wide.u32 %rd4, %r0, 512;
 add.u64 %rd0, %rd0, %rd4;
 mul.wide.u32 %rd4, %r0, 256;
 add.u64 %rd1, %rd1, %rd4;
 add.u64 %rd2, %rd2, %rd4;
 add.u64 %rd3, %rd3, %rd4;
 mad.lo.u32 %r4, %r2, 16, %r3;
 mul.wide.u32 %rd4, %r4, 2;
 add.u64 %rd5, %rd0, %rd4;
 ld.global.u32 %r8, [%rd5];
 ld.global.u32 %r9, [%rd5+256];
 ld.global.u32 %r10, [%rd5+16];
 ld.global.u32 %r11, [%rd5+272];
 mad.lo.u32 %r4, %r3, 8, %r2;
 mul.wide.u32 %rd4, %r4, 2;
 add.u64 %rd5, %rd1, %rd4;
 ld.global.b16 %h0, [%rd5];
 ld.global.b16 %h1, [%rd5+16];
 mov.b32 %r12, {%h0,%h1};
 ld.global.b16 %h2, [%rd5+128];
 ld.global.b16 %h3, [%rd5+144];
 mov.b32 %r13, {%h2,%h3};
 mad.lo.u32 %r4, %r2, 8, %r3;
 mul.wide.u32 %rd4, %r4, 2;
 add.u64 %rd5, %rd2, %rd4;
 ld.global.u32 %r14, [%rd5];
 ld.global.u32 %r15, [%rd5+128];
 mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  {%r16,%r17}, {%r8,%r9,%r10,%r11}, {%r12,%r13}, {%r14,%r15};
 add.u64 %rd5, %rd3, %rd4;
 st.global.u32 [%rd5], %r16;
 st.global.u32 [%rd5+128], %r17;
 ret;
}
)PTX";

#if defined(_MSC_VER)
// MSVC has no _Float16 extension. Convert double<->IEEE binary16 explicitly with
// round-to-nearest-even, matching a native _Float16 cast. This probe compares the
// f16-accumulator MMA against a double sum rounded once to half, so the rounding
// must match IEEE 754 binary16 exactly.
static uint16_t half_bits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 63) << 15);
    int32_t exponent = static_cast<int32_t>((bits >> 52) & 0x7ff);
    uint64_t mantissa = bits & 0x000fffffffffffffULL;
    if (exponent == 0x7ff) return static_cast<uint16_t>(sign | (mantissa ? 0x7e00 : 0x7c00));
    if (exponent == 0) return sign;                       // double subnormal rounds to half zero
    int32_t unbiased = exponent - 1023;
    if (unbiased > 15) return static_cast<uint16_t>(sign | 0x7c00);
    mantissa |= 0x0010000000000000ULL;                    // implicit leading one (53-bit significand)
    const int shift = unbiased >= -14 ? 42 : 28 - unbiased;
    uint64_t rounded = shift >= 64 ? 0 : (mantissa >> shift);
    if (shift < 64) {
        const uint64_t remainder = mantissa & ((1ULL << shift) - 1);
        const uint64_t halfway = 1ULL << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    }
    if (unbiased >= -14) {
        if (rounded >= 0x800) {                            // rounding overflowed into the exponent
            rounded = 0x400;
            if (++unbiased > 15) return static_cast<uint16_t>(sign | 0x7c00);
        }
        return static_cast<uint16_t>(sign | ((unbiased + 15) << 10) | static_cast<uint16_t>(rounded & 0x3ff));
    }
    if (rounded >= 0x400) return static_cast<uint16_t>(sign | 0x0400);  // smallest normal half
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(rounded));
}
static double half_value(uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    double magnitude;
    if (exponent == 0) magnitude = std::ldexp(static_cast<double>(mantissa), -24);
    else if (exponent == 0x1fu) magnitude = mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    else magnitude = std::ldexp(static_cast<double>(mantissa) / 1024.0 + 1.0, static_cast<int>(exponent) - 15);
    return negative ? -magnitude : magnitude;
}
#else
static uint16_t half_bits(double value) {
    _Float16 half = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}
static double half_value(uint16_t bits) {
    _Float16 half;
    std::memcpy(&half, &bits, sizeof(bits));
    return static_cast<double>(half);
}
#endif

int main(int argc, char** argv) {
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_architecture(args.device, props);
        CudaApi cuda(args.cuda_dll);
        cuda.check(cuda.cuInit(0), "cuInit");
        CUdevice device = -1;
        int count = 0;
        cuda.check(cuda.cuDeviceGetCount(&count), "cuDeviceGetCount");
        for (int i = 0; i < count; ++i) {
            CUdevice candidate;
            int bus, pci, domain;
            cuda.check(cuda.cuDeviceGet(&candidate, i), "cuDeviceGet");
            cuda.check(cuda.cuDeviceGetAttribute(&bus, 33, candidate), "PCI bus");
            cuda.check(cuda.cuDeviceGetAttribute(&pci, 34, candidate), "PCI device");
            cuda.check(cuda.cuDeviceGetAttribute(&domain, 50, candidate), "PCI domain");
            if (bus == props.pciBusID && pci == props.pciDeviceID && domain == props.pciDomainID) device = candidate;
        }
        if (device < 0) throw std::runtime_error("CUDA/HIP PCI identity mismatch");
        CUcontext context = nullptr;
        cuda.check(cuda.cuDevicePrimaryCtxRetain(&context, device), "primary context");
        struct Context { CudaApi& api; CUdevice device; ~Context() { api.cuCtxSetCurrent(nullptr); api.cuDevicePrimaryCtxRelease_v2(device); } } context_owner{cuda, device};
        cuda.check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent");
        CUmodule module = nullptr;
        cuda.check(cuda.cuModuleLoadData(&module, source), "MMA PTX JIT");
        struct Module { CudaApi& api; CUmodule module; ~Module() { api.cuModuleUnload(module); } } module_owner{cuda, module};
        CUfunction kernel = nullptr;
        cuda.check(cuda.cuModuleGetFunction(&kernel, module, "d4r_f16_mma_reference"), "MMA kernel");
        constexpr size_t cases = 512;
        std::vector<uint16_t> a(cases * 256), b(cases * 128), c(cases * 128), d(cases * 128);
        uint32_t random = 0x192a321u;
        auto next = [&]() { random = random * 1664525u + 1013904223u; return random; };
        for (size_t n = 0; n < cases; ++n) {
            for (size_t i = 0; i < 256; ++i) a[n * 256 + i] = uint16_t(0x2800u + (next() >> 20));
            for (size_t i = 0; i < 128; ++i) {
                b[n * 128 + i] = uint16_t((next() & 0x8000u) | (0x3000u + (next() >> 20)));
                c[n * 128 + i] = half_bits(n % 3 ? 0. : 0.015625);
            }
            // Sweep large cancellation terms and a tiny surviving product
            // across K positions; this exposes loss hidden by integer fixtures.
            if (n < 256) {
                for (size_t i = 0; i < 256; ++i) a[n * 256 + i] = half_bits(1.);
                std::fill(b.begin() + n * 128, b.begin() + (n + 1) * 128, half_bits(0.));
                for (size_t col = 0; col < 8; ++col) {
                    b[n * 128 + (n % 16) * 8 + col] = half_bits(32768.);
                    b[n * 128 + ((n % 16 + 1) % 16) * 8 + col] = half_bits(-32768.);
                    b[n * 128 + ((n % 16 + 2 + n / 16 % 14) % 16) * 8 + col] = half_bits(0.0009765625);
                }
            }
        }
        struct Allocation {
            CudaApi& api; CUdeviceptr pointer = 0;
            Allocation(CudaApi& api, size_t bytes) : api(api) { api.check(api.cuMemAlloc_v2(&pointer, bytes), "MMA allocate"); }
            ~Allocation() { if (pointer) api.cuMemFree_v2(pointer); }
        } da(cuda, a.size() * 2), db(cuda, b.size() * 2), dc(cuda, c.size() * 2), dd(cuda, d.size() * 2);
        cuda.check(cuda.cuMemcpyHtoD_v2(da.pointer, a.data(), a.size() * 2), "MMA A upload");
        cuda.check(cuda.cuMemcpyHtoD_v2(db.pointer, b.data(), b.size() * 2), "MMA B upload");
        cuda.check(cuda.cuMemcpyHtoD_v2(dc.pointer, c.data(), c.size() * 2), "MMA C upload");
        std::fill(d.begin(), d.end(), uint16_t(0x7e00));
        cuda.check(cuda.cuMemcpyHtoD_v2(dd.pointer, d.data(), d.size() * 2), "MMA NaN sentinel");
        void* parameters[] = {&da.pointer, &db.pointer, &dc.pointer, &dd.pointer};
        cuda.check(cuda.cuLaunchKernel(kernel, cases, 1, 1, 32, 1, 1, 0, nullptr, parameters, nullptr), "MMA launch");
        cuda.check(cuda.cuCtxSynchronize(), "MMA synchronize");
        cuda.check(cuda.cuMemcpyDtoH_v2(d.data(), dd.pointer, d.size() * 2), "MMA download");
        size_t mismatch = 0;
        for (size_t n = 0; n < cases; ++n)
            for (size_t row = 0; row < 16; ++row)
                for (size_t col = 0; col < 8; ++col) {
                    double sum = half_value(c[n * 128 + row * 8 + col]);
                    for (size_t k = 0; k < 16; ++k)
                        sum += half_value(a[n * 256 + row * 16 + k]) * half_value(b[n * 128 + k * 8 + col]);
                    const uint16_t expected = half_bits(sum);
                    const uint16_t actual = d[n * 128 + row * 8 + col];
                    if (expected != actual) {
                        if (mismatch++ < 8) std::fprintf(stderr, "MMA_DIFFERENT case=%zu row=%zu column=%zu expected=%04x actual=%04x\n", n, row, col, expected, actual);
                    }
                }
        std::printf("MMA_REFERENCE cases=%zu elements=%zu differing=%zu\n", cases, d.size(), mismatch);
        if (mismatch) return 5;
        std::printf("PASS CUDA_F16_MMA architecture=gfx1201 cases=%zu exact_f16=1\n", cases);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA_F16_MMA %s\n", error.what());
        loaded_modules();
        return 4;
    }
}
