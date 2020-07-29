#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>
#include <stdio.h>

int const threadsPerBlock = 512;

__global__ void pip_kernel(const int n_boxes, const int n_points, const int box_len,
                           const float *dev_boxes,
                           const float *dev_points, 
                           float *insides) {
  const int cur_box = blockIdx.x;
  const int cur_point = blockIdx.y * threadsPerBlock + threadIdx.x;
  const int cur_thread = blockIdx.x * n_points + cur_point;
  if (cur_point >= n_points || cur_box >= n_boxes) {
      return;
  }
  const float *polygon = dev_boxes +  cur_box * box_len;
  const float cx = dev_points[cur_point * 2 + 0];
  const float cy = dev_points[cur_point * 2 + 1];
  bool inside = false;
  for (int i = 0; i < box_len/2 ; i++ ) {
    const int j = (i == 0 ? box_len/2 - 1 : i - 1);
    if ( (polygon[i*2+1] > cy ) != (polygon[j*2+1] > cy) &&
         cx < (polygon[j*2] - polygon[i*2]) * (cy - polygon[i*2+1]) / (polygon[j*2+1] - polygon[i*2+1]) + polygon[i*2]) {
        inside = !inside;
    }
  }
  insides[cur_thread] = (float) inside;
}

at::Tensor pip_cuda(const at::Tensor boxes, const at::Tensor points) {
  auto insides = at::zeros({boxes.size(0), points.size(0)}, points.options());
  if (insides.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return insides;
  }
  dim3 blocks(boxes.size(0), THCCeilDiv((int)points.size(0), threadsPerBlock));
  dim3 threads(threadsPerBlock);
  pip_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(boxes.size(0),
                                                          points.size(0),
                                                          boxes.size(1),
                                                          boxes.contiguous().data_ptr<float>(),
                                                          points.contiguous().data_ptr<float>(),
                                                          insides.contiguous().data_ptr<float>());

  THCudaCheck(cudaGetLastError());
  return insides;
}
