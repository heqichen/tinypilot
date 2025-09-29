#ifndef __DATADEFS_ML_VISION_H__
#define __DATADEFS_ML_VISION_H__

namespace cooboc {
namespace datadef {
namespace ml {

struct PolicyInput {
    float desire[25 * 8];              // 25 x 8 = 200
    float trafficConvention[2];        // 2
    float lateralControlParams[2];     // 2
    float prevDesiredCurv[25 * 1];     // 25, history of desired curvature
    float featuresBuffer[25 * 512];    // 25 x 512
};

}    // namespace ml
}    // namespace datadef
}    // namespace cooboc


#endif
