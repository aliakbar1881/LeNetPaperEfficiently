#include "../internal/loadpt.hpp"
#include <cstdint>
#include <iostream>
#include <string>

int main() {
std:
  std::string model_path = "/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/"
                      "notebooks/models/lenet5-v2.pt";
  TorchModelPt model(model_path);
  torch::Device device =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  model.to(device);
  auto data_loader =
      torch::data::make_data_loader(torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()));

  for (auto data : *data_loader) {
    auto inputs = data.data;
    auto label = data.target[0].item<int>();
    torch::Tensor resized_img =
        torch::nn::functional::interpolate(
                        inputs, torch::nn::functional::InterpolateFuncOptions()
                                    .size(std::vector<int64_t>{32, 32})
                                    .mode(torch::kNearest));
    int pred = model.predic_class(resized_img);

    std::cout
           << "Predicted : " << pred << "\nActual : " << label << std::endl;
    break;
  }
}

// #include <torch/torch.h>
// #include <torch/script.h>
// #include <iostream>

// int main(){
//         auto net =
//         torch::jit::load("/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/notebooks/models/lenet5-v2.pt");
//         auto data_loader = torch::data::make_data_loader(
//                 torch::data::datasets::MNIST("./data").map(
//                 torch::data::transforms::Stack<>())
//         );

//         torch::Tensor actual_label;
//         torch::Tensor pred;

//         for (auto &img: *data_loader){
//     		auto images = img.data;
//     		std::cout << "Shape" << images.sizes() << std::endl;
//     		torch::Tensor resized_img = torch::nn::functional::interpolate(
//                         images,
//                         torch::nn::functional::InterpolateFuncOptions()
//                             .size(std::vector<int64_t>{32, 32})
//                             .mode(torch::kNearest)
//                     );

//                     pred = net.forward({resized_img}).toTensor();
//                     actual_label = img.target;
//                     break;
//         }

//         // Print nicely: get the actual digit (0-9) instead of the whole
//         tensor std::cout << "Predicted class: " << pred.argmax(1).item<int>()
//                   << " | Actual label: " << actual_label[0].item<int>() <<
//                   std::endl;
// }
