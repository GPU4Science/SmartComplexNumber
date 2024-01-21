#ifndef RUSHCLUSTER_CUDARRAY_CUH
#define RUSHCLUSTER_CUDARRAY_CUH

#include<iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <sstream>

#define TRUST_MODE 1

namespace cudarray_nsp {

enum eCudaArrayPolicy {
    // Set a given block size and ruond robin visit the array
    BULK_ROUND_ROBIN,
    // Split all data evenly to all devices
    EVENLY
};

enum eCudaDeviceStatus {
    // do not use this device
    DEPRECATED,
    // not allocated
    IDLE,
    // alocated but no data on it
    ALLOCATED,
    // holding data
    BUSY
};

template<typename T>
struct cudarray_device_pointer {
    T *data;
    uint64_t offset;
};

template<typename T>
struct cudarray {
    // Allocate a cudarray with given size and device count
    cudarray(uint64_t size, int device_count) : size(size), device_count(device_count), policy(policy) {
        data = new T *[device_count];
        // Count all devices
        policy = eCudaArrayPolicy::EVENLY;
        device_offset = new uint64_t[device_count];
        device_size = new uint64_t[device_count];
        for (int i = 0; i < device_count; i++) {
            data[i] = nullptr;
            device_status[i] = IDLE;
            device_offset[i] = 0;
            device_size[i] = 0;
        }
    }
    // Free all data
    ~cudarray() {
        for (int i = 0; i < device_count; i++) {
            if (data[i] != nullptr) {
                cudaFree(data[i]);
            }
        }
        delete[] data;
        delete[] device_offset;
        delete[] device_size;
    }

    void allocate() {
        if (policy == eCudaArrayPolicy::EVENLY) {
            uint64_t size_per_device = size / device_count;
            for (int i = 0; i < device_count; i++) {
                cudaSetDevice(i);
                uint64_t offset = size_per_device * i;
                uint64_t end_offset = offset + size_per_device;
                uint64_t device_array_size = size_per_device;
                if (end_offset > size) {
                    end_offset = size;
                    device_array_size = end_offset - offset;
                }
                cudaMalloc(&data[i], device_array_size * sizeof(T));
                device_offset[i] = size_per_device * i;
                device_size[i] = device_array_size;
            }
        } else {
            // TODO
        }
    }

    int getDeviceID(uint64_t offset) {
        if (policy == eCudaArrayPolicy::EVENLY) {
            return offset / (size / device_count);
        } else {
            // TODO
        }
    }

    // force inline attribute
    __forceinline__ cudarray_device_pointer<T> getDevicePointer(uint64_t offset) {
        int device_id = getDeviceID(offset);
        cudarray_device_pointer<T> pointer;
        pointer.data = data[device_id];
        pointer.offset = offset - device_offset[device_id];
        return pointer;
    }

    // * Data movement
    int setItem(uint64_t offset, T data) {
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err = cudaMemcpy(pointer.data + pointer.offset, &data, sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }
        return 0;
    }
    T getItem(uint64_t offset) {
        T data;
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err = cudaMemcpy(&data, pointer.data + pointer.offset, sizeof(T), cudaMemcpyDeviceToHost);
        return data;
    }
    int copyFromHostToDevice(T *host_data, uint64_t offset, uint64_t size) {
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err =
            cudaMemcpy(pointer.data + pointer.offset, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }
        return 0;
    }
    int copyFromDeviceToHost(T *host_data, uint64_t offset, uint64_t size) {
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err =
            cudaMemcpy(host_data, pointer.data + pointer.offset, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return -1;
        }
        return 0;
    }

    uint64_t size;
    int device_count;
    eCudaArrayPolicy policy;
    eCudaDeviceStatus *device_status;
    T **data;
    uint64_t *device_offset;
    uint64_t *device_size;
};

} // namespace cudarray_nsp

#endif