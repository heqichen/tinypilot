#ifndef __ML_VISION_RUNNER_H__
#define __ML_VISION_RUNNER_H__

#include <armnn/IRuntime.hpp>
#include <array>
#include "datadefs/ml/vision_output.h"

namespace cooboc {
namespace ml {

class VisionRunner {
  public:
    explicit VisionRunner(const char *tfliteFilepath);
    VisionRunner(const VisionRunner &) = delete;
    VisionRunner &operator=(const VisionRunner &) = delete;
    VisionRunner(VisionRunner &&) = delete;
    VisionRunner &operator=(VisionRunner &&) = delete;
    ~VisionRunner() = default;

    void run(const std::array<std::uint8_t, 12 * 128 * 256> &images,
             const std::array<std::uint8_t, 12 * 128 * 256> &bigImages,
             datadef::ml::VisionOutput &output);

  private:
    armnn::NetworkId networkIdentifier_;
    armnn::BindingPointInfo inputImgsBinding_;
    armnn::BindingPointInfo bigInputImgsBinding_;
    armnn::BindingPointInfo outputBinding_;
};

}    // namespace ml
}    // namespace cooboc

#endif
