#include "../internal/loadjson.hpp"
#include <torch/torch.h>
#include <iostream>
#include <string>

int main() {
    try {
        std::string json_path = "/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/notebooks/models/lenet5-v1.json";
        std::string arch_path = "/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/notebooks/models/lenet5-architecture.txt";

        TorchModelJSON model(json_path, arch_path);

        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        model.to(device);

        auto data_loader = torch::data::make_data_loader(
            torch::data::datasets::MNIST("./data")
                .map(torch::data::transforms::Stack<>())
        );

        torch::NoGradGuard no_grad;

        for (auto& batch : *data_loader) {
            torch::Tensor images = batch.data;
            torch::Tensor labels = batch.target;

            torch::Tensor resized_images = torch::nn::functional::interpolate(
                images,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{32, 32})
                    .mode(torch::kNearest)
            );

            resized_images = resized_images.to(device);
            labels = labels.to(device);

            int predicted = model.predict_class(resized_images);
            int actual = labels[0].item<int>();
            std::cout << "Predicted : " << predicted << "\nActual : " << actual << std::endl;
            break;
        }
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}


// #include "../internal/loadpt.hpp"
// #include <cstdint>
// #include <iostream>
// #include <string>

// int main() {
// std:
//   std::string model_path = "/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/"
//                       "notebooks/models/lenet5-v2.pt";
//   TorchModelPt model(model_path);
//   torch::Device device =
//       torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//   model.to(device);
//   auto data_loader =
//       torch::data::make_data_loader(torch::data::datasets::MNIST("./data").map(
//           torch::data::transforms::Stack<>()));

//   for (auto data : *data_loader) {
//     auto inputs = data.data;
//     auto label = data.target[0].item<int>();
//     torch::Tensor resized_img =
//         torch::nn::functional::interpolate(
//                         inputs, torch::nn::functional::InterpolateFuncOptions()
//                                     .size(std::vector<int64_t>{32, 32})
//                                     .mode(torch::kNearest));
//     int pred = model.predic_class(resized_img);

//     std::cout
//            << "Predicted : " << pred << "\nActual : " << label << std::endl;
//     break;
//   }
// }
