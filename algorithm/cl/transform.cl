
#ifndef TRANSFORMED_WIDTH
#define TRANSFORMED_WIDTH 512
#endif

#ifndef TRANSFORMED_HEIGHT
#define TRANSFORMED_HEIGHT 256
#endif

__kernel void transformY(__global uchar const* const inputFrame,
                         int inputFrameWidth,
                         __constant float* transParameter,
                         __global uchar* const outputFrame) {
    const int tx = get_global_id(0);    // distributed in width direction
    int ox = transParameter[0] * tx + transParameter[1];
    float oyf = transParameter[2] + 0.5F;    // initial y position

    for (int ty = 0; ty < TRANSFORMED_HEIGHT; ++ty) {
        int oy = (int)(oyf);
        outputFrame[ty * TRANSFORMED_WIDTH + tx] = inputFrame[oy * inputFrameWidth + ox];
        oyf += transParameter[0];
    }
}
