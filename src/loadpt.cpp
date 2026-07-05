#include "../internal/loadpt.hpp"
#include <iostream>
#include <string>

TorchModelPt::TorchModelPt(const std::string &model_path) {
  try {
    module_ = torch::jit::load(model_path);
    module_.eval();
    loaded_ = true;
  } catch (const c10::Error &e) {
    loaded_ = false;
    std::cerr << "TorchModelPt can not laod pt model in : " << model_path
              << "\nError: " << e.what() << std::endl;
    throw;
  }
}

void TorchModelPt::to(torch::Device device) {
  if (loaded_) {
    module_.to(device);
  }
}

int TorchModelPt::predic_class(const torch::Tensor &input) {
  torch::Tensor pred = module_.forward({input}).toTensor();
  return pred.argmax(1).item<int>();
}
