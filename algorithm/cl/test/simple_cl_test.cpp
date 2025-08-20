#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <vector>


const char* kernelSource = R"(
__kernel void add_one(__global int* data) {
    int id = get_global_id(0);
    data[id] += 1;
}
)";

int main() {
    const int N = 8;
    std::vector<int> data(N, 0);
    for (int i = 0; i < N; ++i) data[i] = i;

    cl_int err;
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "clGetPlatformIDs failed: " << err << std::endl; return 1; }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "clGetDeviceIDs failed: " << err << std::endl; return 1; }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) { std::cerr << "clCreateContext failed: " << err << std::endl; return 1; }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue || err != CL_SUCCESS) { std::cerr << "clCreateCommandQueue failed: " << err << std::endl; clReleaseContext(context); return 1; }

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, data.data(), &err);
    if (!buffer || err != CL_SUCCESS) { std::cerr << "clCreateBuffer failed: " << err << std::endl; clReleaseCommandQueue(queue); clReleaseContext(context); return 1; }

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (!program || err != CL_SUCCESS) { std::cerr << "clCreateProgramWithSource failed: " << err << std::endl; clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1; }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // 打印编译错误信息
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "clBuildProgram failed: " << err << "\nBuild log:\n" << log.data() << std::endl;
        clReleaseProgram(program); clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "add_one", &err);
    if (!kernel || err != CL_SUCCESS) { std::cerr << "clCreateKernel failed: " << err << std::endl; clReleaseProgram(program); clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1; }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    if (err != CL_SUCCESS) { std::cerr << "clSetKernelArg failed: " << err << std::endl; clReleaseKernel(kernel); clReleaseProgram(program); clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1; }

    size_t global_work_size = N;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "clEnqueueNDRangeKernel failed: " << err << std::endl; clReleaseKernel(kernel); clReleaseProgram(program); clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1; }

    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * N, data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "clEnqueueReadBuffer failed: " << err << std::endl; clReleaseKernel(kernel); clReleaseProgram(program); clReleaseMemObject(buffer); clReleaseCommandQueue(queue); clReleaseContext(context); return 1; }

    for (int i = 0; i < N; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;

    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
