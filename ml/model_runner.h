#ifndef __ML_MODEL_RUNNER_H__
#define __ML_MODEL_RUNNER_H__

namespace cooboc {
namespace ml {
class ModelRunner {
  public:
    explicit ModelRunner(const char *tfliteFilepath);
    ModelRunner(const ModelRunner &) = delete;
    ModelRunner &operator=(const ModelRunner &) = delete;
    ModelRunner(ModelRunner &&) = delete;
    ModelRunner &operator=(ModelRunner &&) = delete;
    ~ModelRunner() = default;
};
}    // namespace ml
}    // namespace cooboc

#endif