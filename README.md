# Infer TF-TRT converted saved model

**Note: currently this example does not work, the aim of the repository is to demonstrate an error with asset file handling using the C++ API of SavedModel.**

## Introduction

This repository is based on the MNIST inference example is based on https://github.com/bmzhao/saved-model-example
The only difference is that after training, we convert the model using the Python API of TF-TRT, and we infer the TRT converted model using the C++ API of SavedModel.

## How to run
```
git clone https://github.com/tfeher/tf-trt-saved-model-example.git saved-model-example
docker run --gpus=all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace -w /workspace nvcr.io/nvidia/tensorflow:21.06-tf2-py3

# Inside the container

cd saved-model-example
mkdir build && cd build
cmake ..
make
```

### Train the model, convert with TF-TRT, and save it
```
python mnist_train.py
```

### Infer the converted model
Get input data
```
cd /workspace/saved-model-example
wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images.idx3-ubyte

```
Run inference
```
TF_CPP_VMODULE=trt_engine_utils=2,trt_engine_op=2,trt_engine_resource_ops=2 ./load_example
```
