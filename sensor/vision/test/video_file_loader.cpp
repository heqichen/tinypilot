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

        // Example: convert to grayscale (process the frame)
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // You can add your own processing here

        // Optional: display the frame
        // cv::imshow("Frame", frame);
        // if (cv::waitKey(1) == 27) break; // Press ESC to exit
    }

    cap.release();
    // cv::destroyAllWindows(); // Uncomment if you use imshow

    std::cout << "Total frames processed: " << frame_count << std::endl;
    return 0;
}
