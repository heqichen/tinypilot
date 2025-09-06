#include "algorithm/cl/transform.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include "algorithm/cl/cl_util.h"

namespace cooboc {
namespace algorithm {
namespace cl {

TransformParameter makeTransformParameter(std::size_t inputWdith,
                                          std::size_t inputHeight,
                                          std::size_t outputWidth,
                                          std::size_t outputHeight) {
    float inputAspect = float(inputWdith) / float(inputHeight);
    float outputAspect = float(outputWidth) / float(outputHeight);
    float scale = (inputAspect > outputAspect) ? (float(inputHeight) / float(outputHeight))
                                               : (float(inputWdith) / float(outputWidth));
    float offsetX = (inputWdith - outputWidth * scale) / 2.0F;
    float offsetY = (inputHeight - outputHeight * scale) / 2.0F;
    return {scale, offsetX, offsetY};
}

void transform(const std::uint8_t *videoFrame,
               const std::size_t inputWidth,
               const std::size_t inputHeight,
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
    cl_kernel transformUVKernel = CL_CHECK_ERR(clCreateKernel(program, "transformUV", &err));
    cl_command_queue queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device, nullptr, &err));

    const std::size_t inputYSize = inputWidth * inputHeight;
    const std::size_t inputUVSize = (inputWidth / 2) * (inputHeight / 2);
    const std::size_t inputSize = inputYSize + (inputUVSize * 2);

    constexpr const std::size_t kOutputWidth = 512;
    constexpr const std::size_t kOutputHeight = 256;

    const std::size_t outputYSize = kOutputWidth * kOutputHeight;
    const std::size_t outputUVSize = (kOutputWidth / 2) * (kOutputHeight / 2);
    const std::size_t outputSize = outputYSize + (outputUVSize * 2);


    cl_mem clInputFrame = CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputSize, (void *)videoFrame, &err));
    cl_mem clOutputBuffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err));


    // Run kernel
    // Y
    cl_int clInputWidth = inputWidth;
    cl_int clInputOffset = 0;
    const TransformParameter transParamY {makeTransformParameter(inputWidth, inputHeight, kOutputWidth, kOutputHeight)};
    const float transformYParameterArray[3] {transParamY.scale, transParamY.offsetX, transParamY.offsetY};
    const cl_mem clTransformYParameter = CL_CHECK_ERR(clCreateBuffer(context,
                                                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                                     sizeof(transformYParameterArray),
                                                                     (void *)transformYParameterArray,
                                                                     &err));
    cl_int clOutputOffset = 0;
    std::printf("Y: scale: %f, offsetX: %f, offsetY: %f\r\n offset input: %u, output: %u\r\n",
                transParamY.scale,
                transParamY.offsetX,
                transParamY.offsetY,
                clInputOffset,
                clOutputOffset);
    CL_CHECK(clSetKernelArg(transformYKernel, 0, sizeof(cl_mem), &clInputFrame));
    CL_CHECK(clSetKernelArg(transformYKernel, 1, sizeof(cl_int), &clInputOffset));
    CL_CHECK(clSetKernelArg(transformYKernel, 2, sizeof(cl_int), &clInputWidth));
    CL_CHECK(clSetKernelArg(transformYKernel, 3, sizeof(cl_mem), &clTransformYParameter));
    CL_CHECK(clSetKernelArg(transformYKernel, 4, sizeof(cl_int), &clOutputOffset));
    CL_CHECK(clSetKernelArg(transformYKernel, 5, sizeof(cl_mem), &clOutputBuffer));

    const std ::size_t transformYWorkSize = kOutputWidth;
    CL_CHECK(
      clEnqueueNDRangeKernel(queue, transformYKernel, 1, nullptr, &transformYWorkSize, nullptr, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    // U
    clInputWidth = inputWidth / 2;
    clInputOffset = inputYSize;
    const float transformUVParameterArray[3] {
      transParamY.scale, transParamY.offsetX / 2.0F, transParamY.offsetY / 2.0F};
    const cl_mem clTransformUVParameter = CL_CHECK_ERR(clCreateBuffer(context,
                                                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                                      sizeof(transformUVParameterArray),
                                                                      (void *)transformUVParameterArray,
                                                                      &err));
    clOutputOffset = outputYSize;
    std::printf("Y: scale: %f, offsetX: %f, offsetY: %f\r\n offset input: %u, output: %u\r\n",
                transformUVParameterArray[0],
                transformUVParameterArray[1],
                transformUVParameterArray[2],
                clInputOffset,
                clOutputOffset);
    CL_CHECK(clSetKernelArg(transformUVKernel, 0, sizeof(cl_mem), &clInputFrame));
    CL_CHECK(clSetKernelArg(transformUVKernel, 1, sizeof(cl_int), &clInputOffset));
    CL_CHECK(clSetKernelArg(transformUVKernel, 2, sizeof(cl_int), &clInputWidth));
    CL_CHECK(clSetKernelArg(transformUVKernel, 3, sizeof(cl_mem), &clTransformUVParameter));
    CL_CHECK(clSetKernelArg(transformUVKernel, 4, sizeof(cl_int), &clOutputOffset));
    CL_CHECK(clSetKernelArg(transformUVKernel, 5, sizeof(cl_mem), &clOutputBuffer));
    const std ::size_t transformUVWorkSize = kOutputWidth / 2;
    CL_CHECK(
      clEnqueueNDRangeKernel(queue, transformUVKernel, 1, nullptr, &transformUVWorkSize, nullptr, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    // V
    clInputOffset = inputYSize + inputUVSize;
    clOutputOffset = outputYSize + outputUVSize;
    std::printf("Y: scale: %f, offsetX: %f, offsetY: %f\r\n offset input: %u, output: %u\r\n",
                transformUVParameterArray[0],
                transformUVParameterArray[1],
                transformUVParameterArray[2],
                clInputOffset,
                clOutputOffset);
    CL_CHECK(clSetKernelArg(transformUVKernel, 0, sizeof(cl_mem), &clInputFrame));
    CL_CHECK(clSetKernelArg(transformUVKernel, 1, sizeof(cl_int), &clInputOffset));
    CL_CHECK(clSetKernelArg(transformUVKernel, 2, sizeof(cl_int), &clInputWidth));
    CL_CHECK(clSetKernelArg(transformUVKernel, 3, sizeof(cl_mem), &clTransformUVParameter));
    CL_CHECK(clSetKernelArg(transformUVKernel, 4, sizeof(cl_int), &clOutputOffset));
    CL_CHECK(clSetKernelArg(transformUVKernel, 5, sizeof(cl_mem), &clOutputBuffer));
    CL_CHECK(
      clEnqueueNDRangeKernel(queue, transformUVKernel, 1, nullptr, &transformUVWorkSize, nullptr, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    clEnqueueReadBuffer(queue, clOutputBuffer, CL_TRUE, 0, outputSize, transformedFrame, 0, nullptr, nullptr);


    clReleaseMemObject(clInputFrame);
    clReleaseMemObject(clOutputBuffer);
    clReleaseMemObject(clTransformYParameter);
    clReleaseMemObject(clTransformUVParameter);
    clReleaseKernel(transformYKernel);
    clReleaseKernel(transformUVKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc