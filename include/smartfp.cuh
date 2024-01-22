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
    /*
    @ brief: This is a struct for smartfp array
    @ param: size: the size of the array (number of complex numbers)
    @ param: level: the level of the array
    @ param: arr: the array
    @ notice!: the size of arr is 2 * size (real part and imaginary part)
    */
    template<typename T>
    struct SmartFPArr {
        int size;
        int level;
        cudarray_nsp::cudarray<T> arr;
    };
} // namespace smartfp_gpu_nsp

