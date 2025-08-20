#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>


int main() {
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

}