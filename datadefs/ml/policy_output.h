#ifndef __DATADEFS_ML_POLICY_OUTPUT_H__
#define __DATADEFS_ML_POLICY_OUTPUT_H__

namespace cooboc {
namespace datadef {
namespace ml {

struct PolicyOutput {
    float plan[4955 - 0];                   // 4955
    float laneLines[5483 - 4955];           // 528
    float laneLinesProb[5491 - 5483];       // 8
    float roadEdges[5755 - 5491];           // 264
    float lead[5857 - 5755];                // 102
    float leadProb[5860 - 5857];            // 3
    float desireState[5868 - 5860];         // 8
    float simPose[5880 - 5868];             // 12
    float desiredCurvature[5882 - 5880];    // 2
    float pad[2];                           // 2
};

}    // namespace ml
}    // namespace datadef
}    // namespace cooboc


#endif
