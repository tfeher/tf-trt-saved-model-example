/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

#include "mnist.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"

// tensorflow::Tensor CreateStringTensor(const std::string &value) {
//   tensorflow::Tensor tensor(tensorflow::DT_STRING,
//   tensorflow::TensorShape({})); tensor.scalar<tensorflow::tstring>()() =
//   value; return tensor;
// }
//
// void AddAssetsTensorsToInputs(
//     const tensorflow::StringPiece export_dir,
//     const std::vector<tensorflow::AssetFileDef> &asset_file_defs,
//     std::vector<std::pair<std::string, tensorflow::Tensor>> *inputs) {
//   if (asset_file_defs.empty()) {
//     return;
//   }
//   for (auto &asset_file_def : asset_file_defs) {
//     tensorflow::Tensor assets_file_path_tensor =
//         CreateStringTensor(tensorflow::io::JoinPath(
//             export_dir, tensorflow::kSavedModelAssetsDirectory,
//             asset_file_def.filename()));
//     inputs->push_back(
//         {asset_file_def.tensor_info().name(), assets_file_path_tensor});
//   }
// }

int main(int argc, char **argv) {
  // Load the saved model from the provided path.
  std::string export_dir = "/data/saved-model-example/mnist_model_trt";
  tensorflow::SavedModelBundle bundle;
  tensorflow::RunOptions run_options;
  tensorflow::Status status =
      tensorflow::LoadSavedModel(tensorflow::SessionOptions(), run_options,
                                 export_dir, {"serve"}, &bundle);
  if (!status.ok()) {
    std::cerr << "Error loading saved model " << status << std::endl;
    return 1;
  }

  // Print the signature defs.
  // This keras model should have an input named
  // "serving_default_flatten_input", and an output named
  // "StatefulPartitionedCall".
  auto signature_map = bundle.GetSignatures();
  for (const auto &name_and_signature_def : signature_map) {
    const auto &name = name_and_signature_def.first;
    const auto &signature_def = name_and_signature_def.second;
    std::cerr << "Name: " << name << std::endl;
    std::cerr << "SignatureDef: " << signature_def.DebugString() << std::endl;
  }

  // Load the MNIST images from the given path.
  std::vector<mnist::MNISTImage> images;
  mnist::MNISTImageReader image_reader(
      "/data/saved-model-example/t10k-images.idx3-ubyte");
  status = image_reader.ReadMnistImages(&images);
  if (!status.ok() || images.empty()) {
    std::cerr << "Error reading MNIST images" << status << std::endl;
    return 2;
  }

  // Convert the first image to a tensorflow::Tensor.
  tensorflow::Tensor input_image = mnist::MNISTImageToTensor(images[0]);
  mnist::MNISTPrint(images[0]);

  tensorflow::Session *session = bundle.GetSession();
  std::vector<tensorflow::Tensor> output_tensors;
  output_tensors.push_back({});

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  inputs.push_back({"serving_default_flatten_input", input_image});
  // std::vector<tensorflow::AssetFileDef> asset_file_defs;
  // for (const auto &asset : bundle.meta_graph_def.asset_file_def()) {
  //   asset_file_defs.push_back(asset);
  // }
  // AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  status =
      session->Run(inputs, {"StatefulPartitionedCall"}, {}, &output_tensors);
  if (!status.ok()) {
    std::cerr << "Error executing session.Run() " << status << std::endl;
    return 3;
  }

  for (const auto &output_tensor : output_tensors) {
    tensorflow::TensorProto proto;
    output_tensor.AsProtoField(&proto);
    std::cerr << "TensorProto Debug Representation: " << proto.DebugString()
              << std::endl;

    const auto &vec = output_tensor.flat_inner_dims<float>();
    float max = 0;
    int argmax = 0;
    for (int i = 0; i < vec.size(); ++i) {
      if (vec(i) > max) {
        argmax = i;
      }
    }
    std::cerr << "Predicted Number: " << argmax << std::endl;
  }
}
