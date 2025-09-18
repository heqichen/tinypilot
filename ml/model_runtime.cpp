#include "ml/model_runtime.h"
#include <armnn/IRuntime.hpp>

namespace cooboc {
namespace ml {

ModelRuntime::ModelRuntime() {
    armnn::IRuntime::CreationOptions options;    // default options
    runtimePtr_ = armnn::IRuntime::Create(options);
}

armnn::IRuntimePtr& ModelRuntime::getRuntime() {
    static ModelRuntime instance;
    return instance.runtimePtr_;
}


}    // namespace ml
}    // namespace cooboc