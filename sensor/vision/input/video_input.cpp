#include "sensor/vision/input/video_input.h"

namespace cooboc {
namespace sensor {
namespace vision {
namespace input {

VideoInput::VideoInput() {}

void VideoInput::registerCallback(const VideoInput::OnVideoFrameCallback callback) {}
bool VideoInput::end() {
    return true;
}
void VideoInput::tick() {}

}    // namespace input
}    // namespace vision
}    // namespace sensor
}    // namespace cooboc
