#ifndef __DATADEFS_ML_VISION_OUTPUT_H__
#define __DATADEFS_ML_VISION_OUTPUT_H__

namespace cooboc {
namespace datadef {
namespace ml {

struct VisionOutput {
    float meta[55 - 0];                        // 55
    float desire_pred[87 - 55];                // 32
    float pose[99 - 87];                       // 12
    float wide_from_device_euler[105 - 99];    // 6
    float road_transform[117 - 105];           // 12
    float hidden_state[632 - 3 - 117];         // 512
    float pad[3];                              // 3
};

}    // namespace ml
}    // namespace datadef
}    // namespace cooboc

#endif