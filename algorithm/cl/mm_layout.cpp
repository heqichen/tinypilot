#include "mm_layout.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace cooboc {
namespace algorithm {
namespace cl {


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


std::string readFile(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


// https://github.com/KhronosGroup/OpenCL-TTL/blob/464e2e14e8e1bc59b74bf922ab6f3dc7c5518d25/opencl/samples/cpp/TTL_sample_runner.cpp#L152

void reorderImageLayout(const std::uint8_t *src, const std::size_t width, const std::size_t height, std::uint8_t *dst) {
    cl_platform_id platform {};
    cl_device_id device {};

    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr));
    cl_context context = CL_CHECK_ERR(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err));
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
        cl_build_status status;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
        std::size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::string logString(logSize + 1U, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, (void *)logString.data(), NULL);
        printf("build failed; status=%d, log: %s", status, logString.c_str());
        clReleaseProgram(program);
        assert(0);
    }

    // Load kernel function in mm_layout.cl
    cl_kernel loadyKernel = CL_CHECK_ERR(clCreateKernel(program, "loadys", &err));
    cl_kernel loaduvKernel = CL_CHECK_ERR(clCreateKernel(program, "loaduv", &err));
    cl_command_queue queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device, nullptr, &err));


    const std::size_t ysize = width * height;
    const std::size_t uvsize = (width / 2) * (height / 2);
    const std::size_t totalSize = ysize + uvsize * 2;

    cl_mem clInputBuffer =
      CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalSize, (void *)src, &err));
    cl_mem clOutputBuffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, totalSize, nullptr, &err));

    // Run kernel
    cl_int clWidth = width;
    cl_int clHeight = height;
    // TODO: Put this 8 division to define, and pass to cl kernel
    constexpr std::size_t kElementSizePerWorker {8U};
    const std::size_t loadYWorkSize = width * height / kElementSizePerWorker;
    CL_CHECK(clSetKernelArg(loadyKernel, 0, sizeof(cl_mem), &clInputBuffer));
    CL_CHECK(clSetKernelArg(loadyKernel, 1, sizeof(cl_mem), &clOutputBuffer));
    CL_CHECK(clSetKernelArg(loadyKernel, 2, sizeof(cl_int), &clWidth));
    CL_CHECK(clSetKernelArg(loadyKernel, 3, sizeof(cl_int), &clHeight));
    CL_CHECK(clEnqueueNDRangeKernel(queue, loadyKernel, 1, nullptr, &loadYWorkSize, nullptr, 0, nullptr, nullptr));

    // U
    cl_int clUOffset = width * height;
    const std::size_t loadUVWorkSize = (width * height) / 4U / kElementSizePerWorker;
    CL_CHECK(clSetKernelArg(loaduvKernel, 0, sizeof(cl_mem), &clInputBuffer));
    CL_CHECK(clSetKernelArg(loaduvKernel, 1, sizeof(cl_mem), &clOutputBuffer));
    CL_CHECK(clSetKernelArg(loaduvKernel, 2, sizeof(cl_int), &clWidth));
    CL_CHECK(clSetKernelArg(loaduvKernel, 3, sizeof(cl_int), &clHeight));
    CL_CHECK(clSetKernelArg(loaduvKernel, 4, sizeof(cl_int), &clUOffset));
    CL_CHECK(clEnqueueNDRangeKernel(queue, loaduvKernel, 1, nullptr, &loadUVWorkSize, nullptr, 0, nullptr, nullptr));

    // V
    cl_int clVOffset = (width * height) + ((width * height) / 4U);
    CL_CHECK(clSetKernelArg(loaduvKernel, 0, sizeof(cl_mem), &clInputBuffer));
    CL_CHECK(clSetKernelArg(loaduvKernel, 1, sizeof(cl_mem), &clOutputBuffer));
    CL_CHECK(clSetKernelArg(loaduvKernel, 2, sizeof(cl_int), &clWidth));
    CL_CHECK(clSetKernelArg(loaduvKernel, 3, sizeof(cl_int), &clHeight));
    CL_CHECK(clSetKernelArg(loaduvKernel, 4, sizeof(cl_int), &clVOffset));
    CL_CHECK(clEnqueueNDRangeKernel(queue, loaduvKernel, 1, nullptr, &loadUVWorkSize, nullptr, 0, nullptr, nullptr));

    // Same as below
    // CL_CHECK(clFinish(queue));
    clEnqueueReadBuffer(queue, clOutputBuffer, CL_TRUE, 0, totalSize, dst, 0, nullptr, nullptr);

    // clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1;
    clReleaseMemObject(clInputBuffer);
    clReleaseMemObject(clOutputBuffer);
    clReleaseKernel(loadyKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc
