import telekinesis
import numpy as np
import time

size = 100000000
arr1 = np.linspace(1.0, 100.0, size)
arr2 = np.linspace(1.0, 100.0, size)

runs = 100
factor = 1.0001

t0 = time.time()
# telekinesis.multiply_with_scalar(arr1, 1.0001, 100)

ptr = telekinesis.array_create(arr1)
for run_index in range(runs):
    telekinesis.array_map(ptr, factor, size)
telekinesis.array_remove(ptr, arr1)

    
print("gpu time: " + str(time.time()-t0))
t0 = time.time()
for _ in range(runs):
    arr2 = arr2 * factor
print("cpu time: " + str(time.time()-t0))

print("results match: " + str(np.allclose(arr1,arr2)))
