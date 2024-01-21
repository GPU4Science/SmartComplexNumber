#include "cudarray.cuh"
#include "telekinesis.cuh"
#include <cstdint>
#include <sys/types.h>

template <typename T>
__global__ void kernel(T *vec, T scalar, int num_elements) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] * scalar;
  }
}

template <typename T> void run_kernel(T *vec, T scalar, int num_elements) {
  dim3 dimBlock(1024, 1, 1);
  dim3 dimGrid(ceil((T)num_elements / dimBlock.x));

  kernel<T><<<dimGrid, dimBlock>>>(vec, scalar, num_elements);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}

template <typename T>
uintptr_t array_create(pybind11::array_t<T> vec) {
  pybind11::buffer_info ha = vec.request();

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "ha.ndim != 1" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  int size = ha.shape[0];
  int size_bytes = size * sizeof(T);
  T *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  T *ptr = reinterpret_cast<T *>(ha.ptr);
  error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  cudaError_t cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }

  return reinterpret_cast<uintptr_t>(gpu_ptr);
}

template <typename T>
void array_map(uintptr_t gpu_ptr_python, T scalar, int size) {
  T *gpu_ptr = reinterpret_cast<T *>(gpu_ptr_python);
  
  run_kernel<T>(gpu_ptr, scalar, size);
}

template <typename T>
void array_remove(uintptr_t gpu_ptr_python, pybind11::array_t<T> vec) {
  pybind11::buffer_info ha = vec.request();

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "ha.ndim != 1" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  int size = ha.shape[0];
  int size_bytes = size * sizeof(T);

  T *gpu_ptr = reinterpret_cast<T *>(gpu_ptr_python);
  T *ptr = reinterpret_cast<T *>(ha.ptr);

  cudaError_t cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }

  cudaError_t error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

PYBIND11_MODULE(telekinesis, m) {
  m.def("array_create", &array_create<double>);
  m.def("array_map", &array_map<double>);
  m.def("array_remove", &array_remove<double>);
}
