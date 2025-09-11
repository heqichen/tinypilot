#include "sensor/vision/input/video_input.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace cooboc {
namespace sensor {
namespace vision {
namespace input {

// Replace "output.mp4" with your video file path
constexpr const char* kVideoPath {"assets/mono_color.mp4"};

VideoInput::VideoInput() :
    callback_ {[](const std::uint8_t*, std::size_t, std::size_t) {
    }},
    isEnd_ {false},
    cap_ {std::string {kVideoPath}} {
    if (!cap_.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << kVideoPath << std::endl;
        throw -1;
    }
}

VideoInput::~VideoInput() {
    cap_.release();
}

void VideoInput::registerCallback(const VideoInput::OnVideoFrameCallback callback) {
    callback_ = callback;
}

bool VideoInput::end() {
    return isEnd_;
}

void VideoInput::tick() {
    cv::Mat frame;

    cap_ >> frame;    // Read next frame
    if (frame.empty()) {
        isEnd_ = true;
        return;    // End of video
    }

    std::size_t width = frame.cols;
    std::size_t height = frame.rows;

    cv::Mat nv12;
    cv::cvtColor(frame, nv12, cv::COLOR_BGR2YUV_I420);

    callback_(nv12.data, width, height);
}
}    // namespace input
}    // namespace vision
}    // namespace sensor
}    // namespace cooboc
