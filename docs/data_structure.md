# Data structure

AOSOA Policy is implement in `src/smartfp.cu`. Traditional Pybind method requires redundant data movement between CPU and GPU. In our library, we use a hack trick to steal the pointer as a `uintptr_t` and pass between our cuda code and python code.

Traditional `pybind` mode:

```text

                                           Cuda Code        
                                   ┌───────────────────────┐
                                   │                       │
                                   │cuda_copy(host->device)│
                                   │cuda_kernel(...)       │
         Python Code               │cuda_copy(device->host)│
┌─────────────────────────┐        │cuda_copy(host->device)│
│xxx.call_cuda_kernel(...)│        │cuda_kernel(...)       │
│xxx.call_cuda_kernel(...)│───────▶│cuda_copy(device->host)│
│xxx.call_cuda_kernel(...)│        │cuda_copy(host->device)│
└─────────────────────────┘        │cuda_kernel(...)       │
                                   │cuda_copy(device->host)│
                                   │...                    │
                                   │                       │
                                   └───────────────────────┘
```

Our method:

```text
                                                     Cuda Code       
                                            ┌───────────────────────┐
           Python Code                      │                       │
┌──────────────────────────────────┐        │                       │
│                                  │        │cuda_allocate()        │
│ptr = xxx.create_data_structure() │        │cuda_copy(host->device)│
│xxx.call_cuda_kernel(...)         │        │cuda_kernel(...)       │
│xxx.call_cuda_kernel(...)         │        │cuda_kernel(...)       │
│xxx.call_cuda_kernel(...)         │───────▶│cuda_kernel(...)       │
│xxx.free_data_structure()         │        │cuda_copy(device->host)│
│                                  │        │cuda_free()            │
│                                  │        │                       │
└──────────────────────────────────┘        │                       │
                                            │                       │
                                            └───────────────────────┘
```

