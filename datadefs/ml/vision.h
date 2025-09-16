#ifndef __DATADEFS_ML_VISION_H__
#define __DATADEFS_ML_VISION_H__

namespace cooboc {
namespace datadef {
namespace ml {

struct Vision {
    float meta[55 - 0];
    float desire_pred[87 - 55];
    float pose[99 - 87];
    float wide_from_device_euler[105 - 99];
    float road_transform[117 - 105];
    float hidden_state[632 - 3 - 117];
    float pad[3];
};

}    // namespace ml
}    // namespace datadef
}    // namespace cooboc

#endif