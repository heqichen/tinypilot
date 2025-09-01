__kernel void project_image(__global const uchar* inBuffer, int width, int height, __global uchar* outBuffer) {
    int out_w = 256;
    int out_h = 128;
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= out_w || y >= out_h)
        return;

    float in_aspect = (float)width / (float)height;
    float out_aspect = (float)out_w / (float)out_h;

    float scale;
    if (in_aspect > out_aspect) {
        // 输入图像更宽，按宽度缩放
        scale = (float)out_w / (float)width;
    } else {
        // 输入图像更高，按高度缩放
        scale = (float)out_h / (float)height;
    }

    // 投影区域在输出图像的中心
    int proj_w = (int)(width * scale + 0.5f);
    int proj_h = (int)(height * scale + 0.5f);
    int offset_x = (out_w - proj_w) / 2;
    int offset_y = (out_h - proj_h) / 2;

    // 判断当前输出像素是否在投影区域内
    if (x < offset_x || x >= offset_x + proj_w || y < offset_y || y >= offset_y + proj_h) {
        // 不产生黑边，直接 return
        return;
    }

    // 反向映射到输入图像坐标
    float in_x = (x - offset_x) / scale;
    float in_y = (y - offset_y) / scale;
    int src_x = (int)(in_x + 0.5f);
    int src_y = (int)(in_y + 0.5f);
    if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
        outBuffer[y * out_w + x] = inBuffer[src_y * width + src_x];
    }
}
