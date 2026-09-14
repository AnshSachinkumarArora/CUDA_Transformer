#include <torch/extension.h>

TORCH_LIBRARY(cuda_transformer, m) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "custom cuda kernels for transformer architecture";
}