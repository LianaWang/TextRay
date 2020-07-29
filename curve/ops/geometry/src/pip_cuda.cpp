#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor pip_cuda(const at::Tensor boxes, const at::Tensor points);

at::Tensor pip(const at::Tensor& boxes, const at::Tensor& points) {
  CHECK_CUDA(boxes);
  CHECK_CUDA(points);
  return pip_cuda(boxes, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pip", &pip, "point-in-polygon");
}