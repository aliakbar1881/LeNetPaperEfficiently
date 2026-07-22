#include "../include/simdjson.h"
#include "../internal/jsonModel.hpp"
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>

static void flatten_array(simdjson::ondemand::value element, std::vector<float>& out) {
    auto array_result = element.get_array();
    if (!array_result.error()) {
        auto arr = array_result.value();
        for (auto child_result : arr) {
            auto child = child_result.value();
            flatten_array(child, out);
        }
        return;
    }

    double value;
    auto err = element.get_double().get(value);
    if (err) {
        throw std::runtime_error("Non‑numeric value encountered (not array, not number)");
    }
    out.push_back(static_cast<float>(value));
}

jsonModel::jsonModel(const std::string& model_json_path){
    auto read_result = simdjson::padded_string::load(model_json_path);
    if (read_result.error()){
        throw std::runtime_error("Faield to open json model file : " + model_json_path);
    }
    json_data_ = std::move(read_result.value());
    getLayerNames();
    json_loaded_ = true;
}

void jsonModel::getLayerNames(){
    auto doc = parser_.iterate(json_data_);
    if (doc.error()){
        throw std::runtime_error("Failed to parse");
    }
    auto obj = doc.get_object();
    for(auto field: obj){
        std::string key(field.unescaped_key().value());
        layer_names.push_back(key);
    }
}

torch::Tensor jsonModel::getLayerWeights(const std::string& layer_name) {
    auto doc = parser_.iterate(json_data_);
    auto obj_result = doc.get_object();
    auto obj = obj_result.value();

    simdjson::ondemand::value found_value;
    bool found = false;

    for (auto field_result : obj_result) {
        if (field_result.error()) continue;
        auto field = field_result.value();
        auto key_result = field_result.unescaped_key();
        if (key_result.error()) continue;
        std::string key(key_result.value());
        bool matches = false;
        if (key == layer_name) {
            matches = true;
        } else if (key.size() > layer_name.size()) {
            char next_char = key[layer_name.size()];
            if ((next_char == '.' || next_char == ' ') && key.compare(0, layer_name.size(), layer_name) == 0) {
                matches = true;
            }
        }

        if (matches) {
            auto value_result = field.value();
            found_value = value_result;
            found = true;
            break;
        }
    }
    std::vector<float> weights;
    flatten_array(found_value, weights);
    return torch::tensor(weights, torch::kFloat32);
}

torch::Tensor jsonModel::getLayerBias(const std::string& layer_name) {
    auto doc = parser_.iterate(json_data_);
    auto obj = doc.get_object().value();
    for (auto field : obj) {
        std::string key(field.unescaped_key().value());
        if (key == layer_name + ".bias" || key.rfind(layer_name + ".bias", 0) == 0) {
            auto value_result = field.value();
            std::vector<float> bias_flat;
            flatten_array(value_result.value(), bias_flat);
            return torch::tensor(bias_flat, torch::kFloat32);
        }
    }
    return torch::Tensor();
}
