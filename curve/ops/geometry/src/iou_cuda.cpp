#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor iou_cuda(const at::Tensor boxes);

at::Tensor iou(const at::Tensor& boxes) {
  CHECK_CUDA(boxes);
  return iou_cuda(boxes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iou", &iou, "iou of polygons");
}
