#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <algorithm>

using namespace std;

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
int const threadsPerBlock = 32;

#define maxn 510
const double eps=1E-8;

__device__ inline int sig(float d){
    return(d>eps)-(d<-eps);
}

__device__ inline int point_eq(const float2 a, const float2 b) {
    return (sig(a.x - b.x) == 0) && (sig(a.y - b.y)==0);
}

__device__ inline void point_swap(float2 *a, float2 *b) {
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2* last)
{
    while ((first!=last)&&(first!=--last)) {
        point_swap (first,last);
        ++first;
    }
}

__device__ inline float cross(float2 o,float2 a,float2 b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}

__device__ inline float area(float2* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}

__device__ inline int lineCross(float2 a,float2 b,float2 c,float2 d,float2&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}

//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
__device__ inline void polygon_cut(float2*p,int&n,float2 a,float2 b, float2* pp){
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!(point_eq(pp[i], pp[i-1])))
            p[n++]=pp[i];
    while(n>1&&point_eq(p[n-1], p[0]))n--;
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a,float2 b,float2 c,float2 d){
    float2 o = make_float2(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    if(s1 == -1) point_swap(&a, &b);
    if(s2 == -1) point_swap(&c, &d);
    float2 p[10]={o,a,b};
    int n=3;
    float2 pp[maxn];
    polygon_cut(p,n,o,c,pp);
    polygon_cut(p,n,c,d,pp);
    polygon_cut(p,n,d,o,pp);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;
}
//求两多边形的交面积
__device__ inline float intersectArea(float2 *ps1, int n1, float2 *ps2, int n2){
    if(area(ps1,n1)<0) point_reverse(ps1,ps1+n1);
    if(area(ps2,n2)<0) point_reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;
}


__device__ inline float devPolyIoU(float const * const pts1, float const * const pts2, 
                                   const int num_pts) {
    float2 ps1[maxn], ps2[maxn];
    for (int i=0; i < num_pts; i++) {
        ps1[i].x = pts1[i*2+0];
        ps1[i].y = pts1[i*2+1];
        ps2[i].x = pts2[i*2+0];
        ps2[i].y = pts2[i*2+1];
    }
    float inter_area = intersectArea(ps1, num_pts, ps2, num_pts);
    float union_area = fabs(area(ps1, num_pts)) + fabs(area(ps2, num_pts)) - inter_area;
    float iou = 0;
    if (union_area == 0) {
        iou = (inter_area + 1) / (union_area + 1);
    } else {
        iou = inter_area / union_area;
    }
    return iou;
}

__global__ void iou_kernel(const float *dev_boxes, int num_box, int num_pts, float *dev_iou){
    const int row_id= blockIdx.x * blockDim.x + threadIdx.x;
    const int col_id= blockIdx.y * blockDim.y + threadIdx.y;
    if(row_id>=num_box || col_id>=num_box) return;
    const float* box1 = dev_boxes + row_id * num_pts * 2;
    const float* box2 = dev_boxes + col_id * num_pts * 2;
    dev_iou[row_id*num_box + col_id] = devPolyIoU(box1, box2, num_pts);
}

at::Tensor iou_cuda(const at::Tensor boxes) {
  auto iou = at::ones({boxes.size(0), boxes.size(0)}, boxes.options());
  const int thread_num = THCCeilDiv((int)boxes.size(0), threadsPerBlock);
  dim3 blocks(thread_num, thread_num);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  iou_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                          boxes.contiguous().data_ptr<float>(),
                                                          (int) boxes.size(0),
                                                          (int) boxes.size(1)/2,
                                                          iou.contiguous().data_ptr<float>());

  THCudaCheck(cudaGetLastError());
  return iou;
}
