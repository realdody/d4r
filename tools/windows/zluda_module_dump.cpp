// Compile one PTX module with the local ZLUDA runtime and keep the resulting AMDGPU code object.
//
// kernels/tex/build_tex.sh uses ZLUDA's offline d4r_emit tool for this. Windows has no d4r_emit
// build here, but the shipped nvcuda.dll performs the same translation at runtime and, with
// D4R_ZLUDA_DUMP_DIR set, writes the compiled module as module.hsaco. This tool only drives that
// path: it loads the module through the CUDA driver API so the runtime compiles it with the
// D4R_ZLUDA_* switches of the environment and drops the code object in the dump directory.
//
// usage: zluda_module_dump <nvcuda.dll> <module.ptx> <kernel>
//
// The PTX and the code object it produces are NVIDIA-derived and must stay local.
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
typedef int (*cuInit_t)(unsigned);
typedef int (*cuDeviceGet_t)(int*, int);
typedef int (*cuCtxCreate_t)(void**, unsigned, int);
typedef int (*cuDevicePrimaryCtxRetain_t)(void**, int);
typedef int (*cuCtxSetCurrent_t)(void*);
typedef int (*cuModuleLoadData_t)(void**, const void*);
typedef int (*cuModuleGetFunction_t)(void**, void*, const char*);

bool read_file(const char* path, std::string& out) {
    FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    char buffer[65536];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) out.append(buffer, read);
    std::fclose(file);
    return true;
}

template <typename T>
bool resolve(HMODULE module, const char* name, T& target) {
    target = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
    if (!target) std::fprintf(stderr, "missing export %s\n", name);
    return target != nullptr;
}
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: zluda_module_dump <nvcuda.dll> <module.ptx> <kernel>\n");
        return 2;
    }
    const char* dll = argv[1];
    const char* ptx_path = argv[2];
    const char* kernel = argv[3];

    std::string ptx;
    if (!read_file(ptx_path, ptx)) {
        std::fprintf(stderr, "cannot read %s\n", ptx_path);
        return 3;
    }
    if (ptx.find('\0') == std::string::npos) ptx.push_back('\0');

    HMODULE library = LoadLibraryA(dll);
    if (!library) {
        std::fprintf(stderr, "LoadLibrary(%s) failed: %lu\n", dll, GetLastError());
        return 4;
    }
    cuInit_t cuInit = nullptr;
    cuDeviceGet_t cuDeviceGet = nullptr;
    cuCtxCreate_t cuCtxCreate = nullptr;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
    cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
    cuModuleLoadData_t cuModuleLoadData = nullptr;
    cuModuleGetFunction_t cuModuleGetFunction = nullptr;
    if (!resolve(library, "cuInit", cuInit) || !resolve(library, "cuDeviceGet", cuDeviceGet) ||
        !resolve(library, "cuModuleLoadData", cuModuleLoadData) ||
        !resolve(library, "cuModuleGetFunction", cuModuleGetFunction))
        return 5;
    resolve(library, "cuCtxCreate", cuCtxCreate);
    resolve(library, "cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain);
    resolve(library, "cuCtxSetCurrent", cuCtxSetCurrent);
    if (!cuDevicePrimaryCtxRetain && !cuCtxCreate) {
        std::fprintf(stderr, "nvcuda.dll exports neither cuDevicePrimaryCtxRetain nor cuCtxCreate\n");
        return 5;
    }

    int status = cuInit(0);
    if (status) { std::fprintf(stderr, "cuInit: %d\n", status); return 6; }
    int device = 0;
    status = cuDeviceGet(&device, 0);
    if (status) { std::fprintf(stderr, "cuDeviceGet: %d\n", status); return 7; }
    void* context = nullptr;
    // The Windows bridge serves the primary context; cuCtxCreate is not supported there.
    if (cuDevicePrimaryCtxRetain) {
        status = cuDevicePrimaryCtxRetain(&context, device);
        if (!status && cuCtxSetCurrent) status = cuCtxSetCurrent(context);
    } else {
        status = cuCtxCreate(&context, 0, device);
    }
    if (status) { std::fprintf(stderr, "context creation: %d\n", status); return 8; }
    void* module = nullptr;
    status = cuModuleLoadData(&module, ptx.c_str());
    if (status) { std::fprintf(stderr, "cuModuleLoadData(%s): %d\n", ptx_path, status); return 9; }
    void* function = nullptr;
    status = cuModuleGetFunction(&function, module, kernel);
    if (status) { std::fprintf(stderr, "cuModuleGetFunction(%s): %d\n", kernel, status); return 10; }
    std::printf("compiled %s: module=%p function=%p\n", kernel, module, function);
    return 0;
}
