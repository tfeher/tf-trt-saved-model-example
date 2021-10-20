cmake_minimum_required(VERSION 3.13)

project(TensorflowSavedModelExample)

set(CMAKE_CXX_STANDARD 14)

#-------------------------------------------------------------
# Configuration
#-------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(TF_INSTALL_DIR "/usr/local" CACHE PATH "Path to Tensorflow install directory")

#-------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------
find_library(tf_shared_lib2 NAMES libtensorflow_cc.so.2 HINTS ${TF_INSTALL_DIR}/lib/tensorflow)
find_library(tf_framework_shared_lib2 NAMES libtensorflow_framework.so.2 HINTS ${TF_INSTALL_DIR}/lib/python3.8/dist-packages/tensorflow)
find_path(trt_include_path NAME NvInfer.h HINTS)
get_filename_component(tf_dir ${tf_shared_lib2} DIRECTORY)
get_filename_component(tf_python_dir ${tf_framework_shared_lib2} DIRECTORY)

set(tf_shared_lib ${tf_dir}/libtensorflow_cc.so)
set(tf_framework_shared_lib ${tf_python_dir}/libtensorflow_framework.so)

add_custom_command(OUTPUT ${tf_framework_shared_lib} ${tf_shared_lib}
  COMMAND ln -s ${tf_shared_lib2} ${tf_shared_lib}
  COMMAND ln -s ${tf_framework_shared_lib2} ${tf_framework_shared_lib}
  COMMENT "Generating legacy symbolic link")

add_custom_target(tf_symlinks DEPENDS ${tf_framework_shared_lib} ${tf_shared_lib})

#-----------------------------------------------------------
# Example Targets
#-----------------------------------------------------------
add_executable(load_example load_example.cc mnist.h mnist.cc)

# Needs symlink libtensorflow_cc.so -> libtensorflow_cc.so
# Means that you have to bazel build libtensorflow_cc.so, not tensorflow_cc
target_link_libraries(load_example tensorflow_cc)
target_link_libraries(load_example tensorflow_framework)
#target_link_libraries(load_example protobuf)

target_compile_options(load_example PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)


target_link_directories(load_example PRIVATE /usr/local/lib/python3.8/dist-packages/tensorflow)
target_link_directories(load_example PRIVATE /usr/local/lib/tensorflow)
# target_link_directories(load_example PRIVATE /root/.cache/bazel/_bazel_root/d5233042e633e2436ecd452ad2f0ea46/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/core/platform)

target_compile_options(load_example PRIVATE -Wl,-rpath=/usr/local/lib/python3.8/dist-packages/tensorflow)

target_include_directories(load_example PRIVATE /usr/local/lib/python3.8/dist-packages/tensorflow/include)
target_include_directories(load_example PRIVATE /data/tensorflow-source)

add_dependencies(load_example tf_symlinks)

# Need to include this, otherwise protobuf's header  #include <google/protobuf/port_def.inc> is not found
# target_include_directories(load_example PRIVATE /usr/local/google/home/bmzhao/Code/tensorflow/bazel-bin/tensorflow/include/src)
