#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CL_CHECK(_expr)                \
    do {                               \
        assert(CL_SUCCESS == (_expr)); \
    } while (0)

#define CL_CHECK_ERR(_expr)               \
    ({                                    \
        cl_int err = CL_INVALID_VALUE;    \
        __typeof__(_expr) _ret = _expr;   \
        assert(_ret &&err == CL_SUCCESS); \
        _ret;                             \
    })


void cl_print_build_errors(cl_program program, cl_device_id device) {
    cl_build_status status;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);

    printf("build failed; status=%d, log: %s", status, log.c_str());
}


std::string readFile(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


// https://github.com/KhronosGroup/OpenCL-TTL/blob/464e2e14e8e1bc59b74bf922ab6f3dc7c5518d25/opencl/samples/cpp/TTL_sample_runner.cpp#L152

int main(int argc, char *argv[], char *envs[]) {
    cl_platform_id platform {};
    cl_device_id device {};

    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr));

    cl_context context = CL_CHECK_ERR(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err));
    cl_command_queue queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device, nullptr, &err));
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7};
    cl_mem buffer = CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.size() * sizeof(int), data.data(), &err));

    // -cl-fast-relaxed-math allows kernels do math on any variable, even wrong data
    // -cl-finite-math-only remove NaN checks
    // -cl-unsafe-math-optimizations remove more checks to sppedup
    // -cl-denorms-are-zero all denorms are flushed to zero
    constexpr const char *clargs =
      "-cl-fast-relaxed-math -cl-finite-math-only -cl-unsafe-math-optimizations -cl-denorms-are-zero ";
    constexpr const char *clSourceCodeFilePath = "mm_layout.cl";


    std::string kernelSource = readFile(clSourceCodeFilePath);
    assert(!kernelSource.empty());
    const char *cKernelSource = kernelSource.c_str();
    cl_program program = CL_CHECK_ERR(clCreateProgramWithSource(context, 1, &cKernelSource, nullptr, &err));
    cl_int err = clBuildProgram(program, 1, &device, clargs, nullptr, nullptr);

    if (err != 0) {
        cl_print_build_errors(program, device);
        assert(0);
    }

    return 0;
}
