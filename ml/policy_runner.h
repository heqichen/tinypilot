#ifndef __ML_POLICY_RUNNER_H__
#define __ML_POLICY_RUNNER_H__


#include <armnn/IRuntime.hpp>
#include <array>
#include "datadefs/ml/policy_input.h"
#include "datadefs/ml/policy_output.h"
#include "datadefs/ml/vision_output.h"

namespace cooboc {
namespace ml {


// {
//  'desire': (1, 100, 8),
//  'traffic_convention': (1, 2),
//  'lateral_control_params': (1, 2),
//  'prev_desired_curv': (1, 100, 1),
//  'features_buffer': (1, 100, 512)
// }


// {
//   'plan': slice(0, 4955, None),
//    'lane_lines': slice(4955, 5483, None),
//    'lane_lines_prob': slice(5483, 5491, None),
//    'road_edges': slice(5491, 5755, None),
//    'lead': slice(5755, 5857, None),
//    'lead_prob': slice(5857, 5860, None),
//    'desire_state': slice(5860, 5868, None),
//    'sim_pose': slice(5868, 5880, None),
//    'desired_curvature': slice(5880, 5882, None),
//    'pad': slice(-2, None, None)
// }


class PolicyRunner {
  public:
    explicit PolicyRunner(const char *tfliteFilepath);
    PolicyRunner(const PolicyRunner &) = delete;
    PolicyRunner &operator=(const PolicyRunner &) = delete;
    PolicyRunner(PolicyRunner &&) = delete;
    PolicyRunner &operator=(PolicyRunner &&) = delete;
    ~PolicyRunner() = default;

    void run(const datadef::ml::PolicyInput &input, datadef::ml::PolicyOutput &output);

  private:
    armnn::NetworkId networkIdentifier_;

    armnn::BindingPointInfo desireBinding_;
    armnn::BindingPointInfo trafficConventionBinding_;
    armnn::BindingPointInfo lateralControlParamsBinding_;
    armnn::BindingPointInfo prevDesiredCurvBinding_;
    armnn::BindingPointInfo featuresBufferBinding_;

    armnn::BindingPointInfo outputBinding_;
};

}    // namespace ml
}    // namespace cooboc

#endif