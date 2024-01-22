#include "cudarray.cuh"
#include "smartfp.cuh"
#include <cstdint>
#include <sys/types.h>

using namespace cudarray_nsp;
using namespace smartfp_gpu_nsp;

template<typename T>
__global__ void vec_mult_kernel(T *vec1, T *vec2, int size, int level) {
    uint64_t group_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t fold_size = 1 << level;

    uint64_t real_index_base = group_index * (2 * fold_size);
    uint64_t imag_index_base = real_index_base + fold_size;

    for (uint64_t idx = 0; idx < fold_size; idx++) {
        ;
        uint64_t real_index = real_index_base + idx;
        uint64_t imag_index = imag_index_base + idx;

        T real1 = vec1[real_index];
        T imag1 = vec1[imag_index];
        T real2 = vec2[real_index];
        T imag2 = vec2[imag_index];

        vec1[real_index] = real1 * real2 - imag1 * imag2;
        vec1[imag_index] = real1 * imag2 + imag1 * real2;
    }
}

template<typename T>
void ExecuteVecMultKernel(SmartFPArr<T> *arr1, SmartFPArr<T> *arr2) {
    int device_count = arr1->arr.device_count;
    int level = arr1->level;
    uint64_t size = arr1->size;
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(ceil((T)size / dimBlock.x));

    for (int device_id = 0; device_id < device_count; device_id++) {
        cudaSetDevice(device_id);
        vec_mult_kernel<T><<<dimGrid, dimBlock>>>(arr1->arr.data[device_id], arr2->arr.data[device_id],
                                                  arr1->arr.device_size[device_id], level);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

/*
@ brief: Create smart FP array
@ param: real, imag: real and imag part of the array
@ param: level: level of AOSOA
@ param: device_count: number of devices
@ return: smart_fp_arr: smart FP array pointer
*/
template<typename T>
uintptr_t SmartFP_Array_Create(pybind11::array_t<T> real, pybind11::array_t<T> imag, int level = 2,
                               int device_count = 1) {
    pybind11::buffer_info real_buffer = real.request();
    pybind11::buffer_info imag_buffer = imag.request();

    // Create smart FP arr pointer
    SmartFPArr<T> *smart_fp_arr = new SmartFPArr<T>();

    // Check if real and imag are the same size
    int real_size = real_buffer.shape[0];
    int imag_size = imag_buffer.shape[0];
    if (real_size != imag_size) {
        std::stringstream strstr;
        strstr << "real_size != imag_size" << std::endl;
        strstr << "real_size: " << real_size << std::endl;
        strstr << "imag_size: " << imag_size << std::endl;
        throw std::runtime_error(strstr.str());
    }

    // Set size and level for AOSOA
    smart_fp_arr->size = real_size;
    smart_fp_arr->level = level;
    smart_fp_arr->arr.resize(real_size * 2, device_count);

    // Convert real and imag pointers to T*
    T *real_ptr = reinterpret_cast<T *>(real_buffer.ptr);
    T *imag_ptr = reinterpret_cast<T *>(imag_buffer.ptr);

    uint64_t index;
    uint64_t fold_size = 1 << level;

    for (index = 0; index < real_size; index += fold_size) {
        // Copy data, AOSOA basic policy
        uint64_t group_id = index / fold_size;
        uint64_t real_index = group_id * (2 * fold_size);
        uint64_t imag_index = real_index + fold_size;
        smart_fp_arr->arr.copyFromHostToDevice(real_ptr + index, real_index, fold_size);
        smart_fp_arr->arr.copyFromHostToDevice(imag_ptr + index, imag_index, fold_size);
    }

    // TODO: Try Copy and then swap in GPU

    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(cuda_status));
    }

    return reinterpret_cast<uintptr_t>(smart_fp_arr);
}

/*
@ brief: Fetch smart FP array
@ param: smartfp_ptr: smart FP array pointer
@ param: real, imag: real and imag part of the array
@ param: level: level of AOSOA
@ param: device_count: number of devices
*/
template<typename T>
void SmartFP_Array_Fetch(uintptr_t smartfp_ptr, pybind11::array_t<T> real, pybind11::array_t<T> imag, int level = 2,
                         int device_count = 1) {
    pybind11::buffer_info real_buffer = real.request();
    pybind11::buffer_info imag_buffer = imag.request();

    // Create smart FP arr pointer
    SmartFPArr<T> *smart_fp_arr = new SmartFPArr<T>();

    // Check if real and imag are the same size
    int real_size = real_buffer.shape[0];
    int imag_size = imag_buffer.shape[0];
    if (real_size != imag_size) {
        std::stringstream strstr;
        strstr << "real_size != imag_size" << std::endl;
        strstr << "real_size: " << real_size << std::endl;
        strstr << "imag_size: " << imag_size << std::endl;
        throw std::runtime_error(strstr.str());
    }

    // Set size and level for AOSOA
    smart_fp_arr->size = real_size;
    smart_fp_arr->level = level;
    smart_fp_arr->arr.resize(real_size, device_count);

    // Convert real and imag pointers to T*
    T *real_ptr = reinterpret_cast<T *>(real_buffer.ptr);
    T *imag_ptr = reinterpret_cast<T *>(imag_buffer.ptr);

    uint64_t index;
    uint64_t fold_size = 1 << level;

    for (index = 0; index < real_size; index += fold_size) {
        // Copy data, AOSOA basic policy
        uint64_t group_id = index / fold_size;
        uint64_t real_index = group_id * (2 * fold_size);
        uint64_t imag_index = real_index + fold_size;
        smart_fp_arr->arr.copyFromDeviceToHost(real_ptr + index, real_index, fold_size);
        smart_fp_arr->arr.copyFromDeviceToHost(imag_ptr + index, imag_index, fold_size);
    }

    // TODO: Try Copy and then swap in GPU

    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(cuda_status));
    }

    return;
}

template<typename T>
void SmartFP_Array_Mul(uintptr_t smart_fp_arr1, uintptr_t smart_fp_arr2) {
    SmartFPArr<T> *smart_fp_arr1_ptr = reinterpret_cast<SmartFPArr<T> *>(smart_fp_arr1);
    SmartFPArr<T> *smart_fp_arr2_ptr = reinterpret_cast<SmartFPArr<T> *>(smart_fp_arr2);

    // Check level and size
    if (smart_fp_arr1_ptr->level != smart_fp_arr2_ptr->level) {
        std::stringstream strstr;
        strstr << "smart_fp_arr1->level != smart_fp_arr2->level" << std::endl;
        strstr << "smart_fp_arr1->level: " << smart_fp_arr1_ptr->level << std::endl;
        strstr << "smart_fp_arr2->level: " << smart_fp_arr2_ptr->level << std::endl;
        throw std::runtime_error(strstr.str());
    }
    if (smart_fp_arr1_ptr->size != smart_fp_arr2_ptr->size) {
        std::stringstream strstr;
        strstr << "smart_fp_arr1->size != smart_fp_arr2->size" << std::endl;
        strstr << "smart_fp_arr1->size: " << smart_fp_arr1_ptr->size << std::endl;
        strstr << "smart_fp_arr2->size: " << smart_fp_arr2_ptr->size << std::endl;
        throw std::runtime_error(strstr.str());
    }

    // Run Kervel
    ExecuteVecMultKernel<T>(smart_fp_arr1_ptr, smart_fp_arr2_ptr);
}

template<typename T>
void SmartFP_Array_Free(uintptr_t smart_fp_arr) {
    SmartFPArr<T> *smart_fp_arr_ptr = reinterpret_cast<SmartFPArr<T> *>(smart_fp_arr);
    delete smart_fp_arr_ptr;
}

PYBIND11_MODULE(smartfp, m) {
    m.def("smartfp_arr_create", &SmartFP_Array_Create<double>);
    m.def("smartfp_arr_compute", &SmartFP_Array_Mul<double>);
    m.def("smartfp_arr_fetch", &SmartFP_Array_Fetch<double>);
    m.def("smartfp_arr_free", &SmartFP_Array_Free<double>);
}
