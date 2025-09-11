#ifndef __SENSOR_VISIONI_INPUT_VISION_INPUT_H__
#define __SENSOR_VISIONI_INPUT_VISION_INPUT_H__

#include <cstdint>
#include <functional>

namespace cooboc {
namespace sensor {
namespace vision {
namespace input {

class VideoInput {
  public:
    using OnVideoFrameCallback = std::function<void(const std::uint8_t *, std::size_t, std::size_t)>;

    VideoInput();
    void registerCallback(const OnVideoFrameCallback callback);
    bool end();
    void tick();

  private:
};

}    // namespace input
}    // namespace vision
}    // namespace sensor
}    // namespace cooboc

#endif
