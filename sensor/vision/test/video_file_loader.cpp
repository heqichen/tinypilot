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
