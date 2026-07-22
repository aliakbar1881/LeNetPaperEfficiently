#include "../internal/loadjson.hpp"
#include "../internal/jsonModel.hpp"
#include "../lib/helper.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

static std::string trim(const std::string& s) {
    auto start = std::find_if_not(s.begin(), s.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    return (start < end) ? std::string(start, end) : std::string();
}

TorchModelJSON::TorchModelJSON(const std::string& json_path, const std::string& arch_path) {
    load_architecture_(arch_path);

    jsonModel model_json(json_path);
    for (auto& layer : architecture_) {
        if (layer.type == "Conv2d" || layer.type == "Linear") {
            layer.weights = model_json.getLayerWeights(layer.name);
            layer.bias = model_json.getLayerBias(layer.name);
        }
    }

    build();
    loaded_ = true;
}

LayerInfo TorchModelJSON::get_layer_info(const std::string& line) {
    LayerInfo info;

    size_t open = line.find('(');
    if (open == std::string::npos) return info;
    size_t close = line.find(')', open + 1);
    if (close == std::string::npos) return info;
    info.name = line.substr(open + 1, close - open - 1);

    size_t colon = line.find(':', close + 1);
    if (colon == std::string::npos) return info;

    size_t type_start = colon + 1;
    while (type_start < line.size() && std::isspace(line[type_start])) type_start++;
    size_t type_end = line.find('(', type_start);
    if (type_end == std::string::npos) return info;
    info.type = line.substr(type_start, type_end - type_start);

    size_t args_open = type_end;
    size_t args_close = line.rfind(')');
    if (args_close == std::string::npos || args_close < args_open) return info;
    info.args = line.substr(args_open + 1, args_close - args_open - 1);

    return info;
}

void TorchModelJSON::load_architecture_(const std::string& architecture_path) {
    std::ifstream file(architecture_path);
    if (!file.is_open()) {
        throw std::runtime_error("Can not open architecture file: " + architecture_path);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue;
        if (trimmed == "LeNet5(" || trimmed == ")") continue;
        LayerInfo info = get_layer_info(trimmed);
        if (!info.name.empty()) {
            architecture_.push_back(info);
        } else {
            std::cout << "Failed to parse: " << trimmed << std::endl;
        }
    }
}

void TorchModelJSON::build() {
    torch::nn::Sequential seq;

    for (const auto& info : architecture_) {
        if (info.type == "Conv2d") {
            seq->push_back(torch::nn::Conv2d(parse_conv2d(info.args)));
        } else if (info.type == "MaxPool2d") {
            seq->push_back(torch::nn::MaxPool2d(parse_maxpool2d(info.args)));
        } else if (info.type == "Flatten") {
            seq->push_back(torch::nn::Flatten());
        } else if (info.type == "Linear") {
            seq->push_back(torch::nn::Linear(parse_linear(info.args)));
        } else {
            throw std::runtime_error("Unsupported layer type: " + info.type);
        }
    }

    model_ = seq;

    size_t idx = 0;
    for (const auto& info : architecture_) {
        if (info.type == "Conv2d") {
            auto* conv = model_[idx]->as<torch::nn::Conv2d>();
            if (conv && info.weights.numel() > 0) {
                conv->weight.set_data(info.weights.reshape(conv->weight.sizes()));
            }
            if (conv && info.bias.numel() > 0) {
                conv->bias.set_data(info.bias.reshape(conv->bias.sizes()));
            }
        } else if (info.type == "Linear") {
            auto* linear = model_[idx]->as<torch::nn::Linear>();
            if (linear && info.weights.numel() > 0) {
                linear->weight.set_data(info.weights.reshape(linear->weight.sizes()));
            }
            if (linear && info.bias.numel() > 0) {
                linear->bias.set_data(info.bias.reshape(linear->bias.sizes()));
            }
        }
        idx++;
    }
}

void TorchModelJSON::to(torch::Device device) {
    model_->to(device);
}

torch::Tensor TorchModelJSON::forward(const torch::Tensor& input) {
    return model_->forward(input);
}

int TorchModelJSON::predict_class(const torch::Tensor& input) {
    torch::Tensor pred = forward(input);
    return pred.argmax(1).item<int>();
}
