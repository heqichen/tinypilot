#ifndef __SENSOR_VISIONI_INPUT_VISION_INPUT_H__
#define __SENSOR_VISIONI_INPUT_VISION_INPUT_H__

#include <cstdint>
#include <functional>
#include <opencv2/opencv.hpp>


namespace cooboc {
namespace sensor {
namespace vision {
namespace input {


class VideoInput {
  public:
    using OnVideoFrameCallback = std::function<void(const std::uint8_t *, std::size_t, std::size_t)>;

    VideoInput(const char *videoPath);
    virtual ~VideoInput();
    void registerCallback(const OnVideoFrameCallback callback);
    bool end();
    void tick();

  private:
    OnVideoFrameCallback callback_;
    bool isEnd_;
    cv::VideoCapture cap_;
};

}    // namespace input
}    // namespace vision
}    // namespace sensor
}    // namespace cooboc

#endif
