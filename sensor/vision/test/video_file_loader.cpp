#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Replace "output.mp4" with your video file path
    std::string video_path = "mono_color.mp4";
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
        return -1;
    }

    int frame_count = 0;
    cv::Mat frame;
    while (true) {
        cap >> frame; // Read next frame
        if (frame.empty()) {
            break; // End of video
        }
        frame_count++;
        std::cout << "Processing frame " << frame_count
                  << " - size: " << frame.cols << "x" << frame.rows << std::endl;
        // 输出颜色空间信息
        int channels = frame.channels();
        std::string color_space = (channels == 3) ? "BGR (3 channels)" : (channels == 1) ? "Grayscale (1 channel)" : "Unknown";
        std::cout << "Frame channels: " << channels << " -> Color space: " << color_space << std::endl;

        // Example: convert to RGB (process the frame)
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        // 输出RGB通道信息
        std::cout << "After conversion: channels = " << rgb.channels() << " (RGB)" << std::endl;

        // 输出第3行第5列像素的RGB值（注意OpenCV的行列索引从0开始）
        int row = 2, col = 4;
        if (rgb.rows > row && rgb.cols > col) {
            cv::Vec3b pixel = rgb.at<cv::Vec3b>(row, col);
            std::cout << "Pixel at (3,5): R=" << (int)pixel[0] << ", G=" << (int)pixel[1] << ", B=" << (int)pixel[2] << std::endl;
        } else {
            std::cout << "Frame too small for pixel (3,5)" << std::endl;
        }

        // 将RGB矩阵转换为YUV_I420格式
        cv::Mat yuv;
        cv::cvtColor(rgb, yuv, cv::COLOR_RGB2YUV_I420);
        std::cout << "After conversion to YUV_I420: size = " << yuv.cols << "x" << yuv.rows << ", channels = " << yuv.channels() << std::endl;

        // // 输出YUV_I420矩阵第一行的所有像素数据（Y分量）
        // if (yuv.rows > 0) {
        //     std::cout << "YUV_I420 first row (Y plane): ";
        //     int width = yuv.cols;
        //     for (int i = 0; i < width; ++i) {
        //         std::cout << (int)yuv.at<uchar>(0, i) << " ";
        //     }
        //     std::cout << std::endl;
        // } else {
        //     std::cout << "YUV frame too small for first row" << std::endl;
        // }
/*
YUV_I420（也叫 YUV420p）矩阵的内存排布如下：

Y分量（亮度）：前面一大块，大小为宽 × 高，每个像素一个字节。
U分量（色度，Cb）：紧接着Y分量，大小为 (宽/2) × (高/2)，每个像素一个字节。
V分量（色度，Cr）：最后一块，大小同U分量，也是 (宽/2) × (高/2)，每个像素一个字节。
整体内存布局是：

其中 Y 是每个像素，U/V 是每2×2像素共用一个值（即采样率为 1/4）。

举例：假设图像为 640×480

Y: 640×480 字节
U: 320×240 字节
V: 320×240 字节
总内存 = 640×480 + 320×240 + 320×240 字节

在 OpenCV 的 cv::Mat 中，所有数据是连续存储的，可以通过 yuv.data 指针访问。
*/

        // 验证YUV_I420内存排布：输出Y、U、V分量的起始地址和部分数据
        int width = yuv.cols / 1.5;
        int height = yuv.rows;
        int y_size = width * height;
        int uv_width = width / 2;
        int uv_height = height / 2;
        int uv_size = uv_width * uv_height;
        uchar* y_ptr = yuv.data;
        uchar* u_ptr = yuv.data + y_size;
        uchar* v_ptr = yuv.data + y_size + uv_size;
        std::cout << "Y plane start: " << (void*)y_ptr << ", U plane start: " << (void*)u_ptr << ", V plane start: " << (void*)v_ptr << std::endl;
        std::cout << "First 8 Y values: ";
        for (int i = 0; i < 8 && i < y_size; ++i) std::cout << (int)y_ptr[i] << " ";
        std::cout << "\nFirst 8 U values: ";
        for (int i = 0; i < 8 && i < uv_size; ++i) std::cout << (int)u_ptr[i] << " ";
        std::cout << "\nFirst 8 V values: ";
        for (int i = 0; i < 8 && i < uv_size; ++i) std::cout << (int)v_ptr[i] << " ";
        std::cout << std::endl;

        
        
        std::cout << std::endl;
    }

    cap.release();
    // cv::destroyAllWindows(); // Uncomment if you use imshow

    std::cout << "Total frames processed: " << frame_count << std::endl;
    return 0;
}
