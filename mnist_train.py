# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.input_shape

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

model.save('mnist_model')

print("Model trained and saved under mnist_model")


def trt_convert(input_path, output_path, input_shapes=None, explicit_batch=False,
                dtype=np.float32, precision='FP32'):
    conv_params=trt.TrtConversionParams(
        precision_mode=precision, minimum_segment_size=3,
        max_workspace_size_bytes=2*1<<30, maximum_cached_engines=1)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_path, conversion_params=conv_params,
        use_dynamic_shape=explicit_batch,
        dynamic_shape_profile_strategy="Optimal")

    converter.convert()

    def input_fn():
        for shapes in input_shapes:
            # return a list of input tensors
            yield [np.ones(shape=x).astype(dtype) for x in shapes]

    if input_shapes is not None:
        converter.build(input_fn)
    converter.save(output_path)

trt_convert('mnist_model', 'mnist_model_trt', input_shapes=[[(1, 28, 28)]], explicit_batch=True)

print("Model converted and saved under mnist_model_trt")
