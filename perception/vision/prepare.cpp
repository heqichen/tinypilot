#include "perception/vision/prepare.h"
#include <array>
#include <cstdint>
#include "algorithm/cl/transform.h"

namespace cooboc {
namespace perception {
namespace vision {

void prepare(std::uint8_t const* const video_frame,
             std::size_t const width,
             std::size_t const height,
             std::uint8_t* const image_data) {
    const std::size_t inputYSize = width * height;
    const std::size_t inputUVSize = width * height / 4U;

    const std::size_t outputYSize = 512U * 256U;
    const std::size_t outputUVSize = 512U * 256U / 4U;

    // Transfrom big image into small
    const algorithm::cl::TransformParameter transParamY {
      algorithm::cl::makeTransformParameter(width, height, 512U, 256U)};
    const algorithm::cl::TransformParameter transParamUV {
      transParamY.scale, transParamY.offsetX / 2.0F, transParamY.offsetY / 2.0F};

    std::array<std::uint8_t, 512U * 256U * 3U / 2U> resizedFrame {};
    // Y
    algorithm::cl::transform(video_frame, width, height, transParamY, resizedFrame.data());
    // U
    algorithm::cl::transform(
      video_frame + inputYSize, width / 2U, height / 2U, transParamUV, resizedFrame.data() + outputYSize);
    // V
    algorithm::cl::transform(video_frame + inputYSize + inputUVSize,
                             width / 2U,
                             height / 2U,
                             transParamUV,
                             resizedFrame.data() + outputYSize + outputUVSize);
}
}    // namespace vision
}    // namespace perception
}    // namespace cooboc