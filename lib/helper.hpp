#ifndef HELPER_HPP
#define HELPER_HPP

#include <regex>
#include <vector>
#include <string>
#include <torch/torch.h>

static std::vector<int64_t> extract_all_ints(const std::string& s) {
    std::vector<int64_t> nums;
    std::regex num_pattern(R"(\d+)");
    auto begin = std::sregex_iterator(s.begin(), s.end(), num_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        nums.push_back(std::stoll(it->str()));
    }
    return nums;
}

static int64_t extract_named_int(const std::string& args, const std::string& key) {
    std::regex pattern(key + R"(=\(?(\d+))");
    std::smatch match;
    if (std::regex_search(args, match, pattern)) {
        return std::stoll(match[1].str());
    }
    return -1;
}

static torch::nn::Conv2dOptions parse_conv2d(const std::string& args) {
    auto nums = extract_all_ints(args);
    int64_t in_channels = (nums.size() > 0) ? nums[0] : 1;
    int64_t out_channels = (nums.size() > 1) ? nums[1] : 6;
    int64_t kernel_size = extract_named_int(args, "kernel_size");
    if (kernel_size == -1) kernel_size = (nums.size() > 2) ? nums[2] : 5;
    int64_t stride = extract_named_int(args, "stride");
    if (stride == -1) stride = (nums.size() > 3) ? nums[3] : 1;
    return torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(0);
}

static torch::nn::LinearOptions parse_linear(const std::string& args) {
    auto nums = extract_all_ints(args);
    int64_t in_features = (nums.size() > 0) ? nums[0] : 0;
    int64_t out_features = (nums.size() > 1) ? nums[1] : 0;
    return torch::nn::LinearOptions(in_features, out_features);
}

static torch::nn::MaxPool2dOptions parse_maxpool2d(const std::string& args) {
    int64_t kernel_size = extract_named_int(args, "kernel_size");
    int64_t stride = extract_named_int(args, "stride");
    if (kernel_size == -1) kernel_size = 2;
    if (stride == -1) stride = 1;
    return torch::nn::MaxPool2dOptions(kernel_size).stride(stride);
}
#endif
