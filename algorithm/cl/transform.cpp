#include "algorithm/cl/transform.h"
#include <cstdint>
#include "algorithm/cl/cl_util.h"

namespace cooboc {
namespace algorithm {
namespace cl {


void transform(const std::uint8_t *videoFrame,
               const std::size_t width,
               const std::size_t height,
               const TransformParameter &transformParameter,
               std::uint8_t *transformedFrame) {
    cl_platform_id platform {};
    cl_device_id device {};

    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr));
    cl_context context = CL_CHECK_ERR(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err));
    // -cl-fast-relaxed-math allows kernels do math on any variable, even wrong data
    // -cl-finite-math-only remove NaN checks
    // -cl-unsafe-math-optimizations remove more checks to sppedup
    // -cl-denorms-are-zero all denorms are flushed to zero
    char clargs[1024];
    snprintf(clargs,
             1023,
             "-cl-fast-relaxed-math "
             "-cl-finite-math-only "
             "-cl-unsafe-math-optimizations "
             "-cl-denorms-are-zero "
             "-DTRANSFORMED_WIDTH=%d "
             "-DTRANSFORMED_HEIGHT=%d ",
             512,
             256);    // Put constant value to parameter


    constexpr const char *clSourceCodeFilePath = "transform.cl";
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

    cl_kernel transformYKernel = CL_CHECK_ERR(clCreateKernel(program, "transformY", &err));
    cl_command_queue queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device, nullptr, &err));

    const std::size_t inputYSize = width * height;
    const std::size_t inputUVSize = (width / 2) * (height / 2);
    const std::size_t inputSize = inputYSize + (inputUVSize * 2);

    constexpr const std::size_t kOutputWidth = 512;
    constexpr const std::size_t kOutputHeight = 256;

    const std::size_t outputYSize = kOutputWidth * kOutputHeight;
    const std::size_t outputUVSize = (kOutputWidth / 2) * (kOutputHeight / 2);
    const std::size_t outputSize = outputYSize + (outputUVSize * 2);

    const float transformParameterArray[3] {
      transformParameter.scale, transformParameter.offsetX, transformParameter.offsetY};

    cl_mem clInputFrame = CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputSize, (void *)videoFrame, &err));
    cl_mem clOutputBuffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err));
    cl_mem clTransformParameter = CL_CHECK_ERR(clCreateBuffer(context,
                                                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                              sizeof(transformParameterArray),
                                                              (void *)transformParameterArray,
                                                              &err));


    const std ::size_t transformYWorkSize = kOutputWidth;

    // Run kernel
    cl_int clInputWidth = width;
    CL_CHECK(clSetKernelArg(transformYKernel, 0, sizeof(cl_mem), &clInputFrame));
    CL_CHECK(clSetKernelArg(transformYKernel, 1, sizeof(cl_int), &clInputWidth));
    CL_CHECK(clSetKernelArg(transformYKernel, 2, sizeof(cl_mem), &clTransformParameter));
    CL_CHECK(clSetKernelArg(transformYKernel, 3, sizeof(cl_mem), &clOutputBuffer));
    CL_CHECK(
      clEnqueueNDRangeKernel(queue, transformYKernel, 1, nullptr, &transformYWorkSize, nullptr, 0, nullptr, nullptr));

    clEnqueueReadBuffer(queue, clOutputBuffer, CL_TRUE, 0, outputSize, transformedFrame, 0, nullptr, nullptr);

    // clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1;
    clReleaseMemObject(clInputFrame);
    clReleaseMemObject(clOutputBuffer);
    clReleaseKernel(transformYKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc