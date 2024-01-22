# SmartComplexNumber

Utilize AOSOA to enhance computational performance in complex number operations.

# What is AOSOA?

> Array of structures of arrays (AoSoA) or tiled array of structs is a hybrid approach between the previous layouts, in which data for different fields is interleaved using tiles or blocks with size equal to the SIMD vector size.
>
> -- Wikipedia: AoS and SoA

```c
struct point3Dx8 {
    float x[8];
    float y[8];
    float z[8];
};
struct point3Dx8 points[(N+7)/8];
float get_point_x(int i) { return points[i/8].x[i%8]; }
```

# Prerequisites

Cuda installed in /usr/local/cuda 

Python 3.6 - 3.10 (Strongly suggest 3.10.3-3.10.12)

Cmake 3.6 or greater

# Installation

```bash
bash install.bash
```

Test it with 

```bash
python3 test/test.py
``` 

# Refernces

- https://en.wikipedia.org/wiki/AoS_and_SoA
- https://github.com/torstem/demo-cuda-pybind11
- https://github.com/PWhiddy/pybind11-cuda

Give an example of how to pass (host/device) pointers between C++ and python.
