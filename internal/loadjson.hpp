#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <vector>

struct LayerInfo {
    std::string name;
    std::string type;
    std::string args;
    torch::Tensor weights;
    torch::Tensor bias;
};

class TorchModelJSON {
public:
    explicit TorchModelJSON(const std::string& json_path, const std::string& txt_path);
    int predict_class(const torch::Tensor& inputs);
    void to(torch::Device device);
    void build();
    torch::Tensor forward(const torch::Tensor& inputs);
private:
    torch::nn::Sequential model_;
    bool loaded_ = false;
    std::vector<LayerInfo> architecture_;
    void load_architecture_(const std::string& architecture_path);
    LayerInfo get_layer_info(const std::string& model_arch);
};