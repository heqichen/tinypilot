
#ifndef TRANSFORMED_WIDTH
#define TRANSFORMED_WIDTH 512
#endif

#ifndef TRANSFORMED_HEIGHT
#define TRANSFORMED_HEIGHT 256
#endif

__kernel void transformY(__global uchar const* const inputFrame, __global uchar* const outputFrame) {
    const int x = get_global_id(0);    // distributed in width direction
    for (int y = 0; y < TRANSFORMED_HEIGHT; ++y) {
        outputFrame[y * TRANSFORMED_WIDTH + x] = x;
    }
}
