#ifndef __ML_MODEL_RUNTIME_H__
#define __ML_MODEL_RUNTIME_H__

#include <armnn/IRuntime.hpp>

namespace cooboc {
namespace ml {

class ModelRuntime {
  public:
    static armnn::IRuntimePtr& getRuntime();

  private:
    ModelRuntime();
    ModelRuntime(const ModelRuntime&) = delete;
    ModelRuntime& operator=(const ModelRuntime&) = delete;
    ModelRuntime(ModelRuntime&&) = delete;
    ModelRuntime& operator=(ModelRuntime&&) = delete;

    armnn::IRuntimePtr runtimePtr_ {nullptr, nullptr};
};

}    // namespace ml
}    // namespace cooboc

#endif
