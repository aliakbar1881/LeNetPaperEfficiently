#ifndef LOADPT_HPP
#define LOADPT_HPP

#include <string>
#include <torch/script.h>
#include <torch/torch.h>

class TorchModelPt {
public:
  explicit TorchModelPt(const std::string &model_path);
  void to(torch::Device deivce);
  int predic_class(const torch::Tensor &input);

private:
  bool loaded_ = false;
  torch::jit::script::Module module_;
};
#endif
