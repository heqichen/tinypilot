#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>

int main() {
    // Step 1: Get number of OpenCL platforms
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    // Step 2: Get the first available platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // Step 3: Get number of GPU devices available on the platform
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);

    // Step 4: Get the first GPU device ID
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Step 5: Define buffers to store device info
    char name[128], vendor[128];
    cl_ulong mem;
    cl_uint units;

    // Step 6: Retrieve device name, vendor, memory size, and compute units
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);

    // Step 7: Print the collected device information
    std::cout << "Device Name: " << name << "\n";
    std::cout << "Vendor: " << vendor << "\n";
    std::cout << "Memory: " << mem / (1024 * 1024) << " MB\n";
    std::cout << "Compute Units: " << units << "\n";

    return 0;
}

