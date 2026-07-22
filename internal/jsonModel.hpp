#pragma once

#include "../include/simdjson.h"
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>


class jsonModel{
    public:
        explicit jsonModel(const std::string& model_json_path);
        torch::Tensor getLayerWeights(const std::string& layer_name);
        torch::Tensor getLayerBias(const std::string& layer_name);
        std::vector<std::string> layer_names;
    private:
        simdjson::ondemand::parser parser_;
        simdjson::padded_string json_data_;
        bool json_loaded_ = false;
        void getLayerNames();
};