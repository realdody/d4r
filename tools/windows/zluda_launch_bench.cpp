// Measure the per-launch host cost of the ZLUDA/CUDA driver path, independent of DLSS.
//
// Loads a trivial PTX kernel through the local nvcuda.dll and times repeated
// cuLaunchKernel calls, so the cost measured is the driver-side launch path
// (module lookup, stream bookkeeping, HIP dispatch) with no DLSS or network in
// the way. Usage: zluda_launch_bench <nvcuda.dll> <launches>
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
typedef int (*cuInit_t)(unsigned);
typedef int (*cuDeviceGet_t)(int*, int);
typedef int (*cuDevicePrimaryCtxRetain_t)(void**, int);
typedef int (*cuCtxSetCurrent_t)(void*);
typedef int (*cuModuleLoadData_t)(void**, const void*);
typedef int (*cuModuleGetFunction_t)(void**, void*, const char*);
typedef int (*cuMemAlloc_t)(unsigned long long*, size_t);
typedef int (*cuLaunchKernel_t)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, unsigned, void*, void**, void**);
typedef int (*cuCtxSynchronize_t)(void);
typedef int (*cuMemcpyDtoH_t)(void*, unsigned long long, size_t);

template <typename T>
bool resolve(HMODULE module, const char* name, T& target) {
    target = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
    if (!target) std::fprintf(stderr, "missing export %s\n", name);
    return target != nullptr;
}

// One store, so the launch is real but trivial.
const char kPtx[] =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry d4r_bench_kernel(.param .u64 d4r_bench_out)\n"
    "{\n"
    "    .reg .u64 %rd<2>;\n"
    "    .reg .u32 %r<2>;\n"
    "    ld.param.u64 %rd1, [d4r_bench_out];\n"
    "    mov.u32 %r1, 1;\n"
    "    st.global.u32 [%rd1], %r1;\n"
    "    ret;\n"
    "}\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: zluda_launch_bench <nvcuda.dll> <launches>\n");
        return 2;
    }
    const int launches = std::atoi(argv[2]);
    if (launches < 10) { std::fprintf(stderr, "launches must be >= 10\n"); return 2; }

    HMODULE library = LoadLibraryA(argv[1]);
    if (!library) { std::fprintf(stderr, "LoadLibrary failed: %lu\n", GetLastError()); return 3; }
    cuInit_t cuInit = nullptr; cuDeviceGet_t cuDeviceGet = nullptr;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr; cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
    cuModuleLoadData_t cuModuleLoadData = nullptr; cuModuleGetFunction_t cuModuleGetFunction = nullptr;
    cuMemAlloc_t cuMemAlloc = nullptr; cuLaunchKernel_t cuLaunchKernel = nullptr;
    cuCtxSynchronize_t cuCtxSynchronize = nullptr;
    cuMemcpyDtoH_t cuMemcpyDtoH = nullptr;
    if (!resolve(library, "cuInit", cuInit) || !resolve(library, "cuDeviceGet", cuDeviceGet) ||
        !resolve(library, "cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain) ||
        !resolve(library, "cuCtxSetCurrent", cuCtxSetCurrent) || !resolve(library, "cuModuleLoadData", cuModuleLoadData) ||
        !resolve(library, "cuModuleGetFunction", cuModuleGetFunction) || !resolve(library, "cuMemAlloc_v2", cuMemAlloc) ||
        !resolve(library, "cuLaunchKernel", cuLaunchKernel) || !resolve(library, "cuCtxSynchronize", cuCtxSynchronize) ||
        !resolve(library, "cuMemcpyDtoH_v2", cuMemcpyDtoH))
        return 4;

    if (cuInit(0)) { std::fprintf(stderr, "cuInit failed\n"); return 5; }
    int device = 0; if (cuDeviceGet(&device, 0)) { std::fprintf(stderr, "cuDeviceGet failed\n"); return 5; }
    void* context = nullptr;
    if (cuDevicePrimaryCtxRetain(&context, device) || cuCtxSetCurrent(context)) {
        std::fprintf(stderr, "context failed\n"); return 5;
    }
    unsigned long long buffer = 0;
    if (cuMemAlloc(&buffer, 4)) { std::fprintf(stderr, "cuMemAlloc failed\n"); return 5; }
    void* module = nullptr;
    if (cuModuleLoadData(&module, kPtx)) { std::fprintf(stderr, "cuModuleLoadData failed\n"); return 5; }
    void* function = nullptr;
    if (cuModuleGetFunction(&function, module, "d4r_bench_kernel")) {
        std::fprintf(stderr, "cuModuleGetFunction failed\n"); return 5;
    }

    void* arguments[] = {&buffer};
    auto launch = [&]() { return cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr); };
    if (launch()) { std::fprintf(stderr, "warmup launch failed\n"); return 6; }
    if (cuCtxSynchronize()) { std::fprintf(stderr, "warmup sync failed\n"); return 6; }
    for (int i = 0; i < 64; ++i) (void)launch();
    (void)cuCtxSynchronize();

    // Unsynchronised: the host-side dispatch cost, which is what a DLSS frame pays
    // between its (separately synchronised) stages.
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < launches; ++i) (void)launch();
    const auto issued = std::chrono::steady_clock::now();
    (void)cuCtxSynchronize();
    const auto completed = std::chrono::steady_clock::now();

    const double issueUs = std::chrono::duration<double, std::micro>(issued - begin).count() / launches;
    const double totalUs = std::chrono::duration<double, std::micro>(completed - begin).count() / launches;

    // Verify the launches really executed: the kernel stores 1.
    unsigned value = 0;
    if (cuMemcpyDtoH(&value, buffer, sizeof(value))) {
        std::fprintf(stderr, "cuMemcpyDtoH failed\n");
        return 7;
    }
    std::printf("launches=%d issue_us_per_launch=%.2f round_trip_us_per_launch=%.2f output=%u (%s)\n",
        launches, issueUs, totalUs, value, value == 1u ? "executed" : "NOT EXECUTED");
    return value == 1u ? 0 : 8;
}