#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "cudarray.cuh"

using namespace cudarray_nsp;

namespace smartfp_gpu_nsp {
    template<typename T>
    struct SmartFPArr {
        int size;
        int level;
        cudarray_nsp::cudarray<T> arr;
    };
} // namespace smartfp_gpu_nsp

