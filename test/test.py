import smartfp
import numpy as np
import time

size = 1 << 21
arr1_real = np.linspace(1.0, 1.0, size)
arr1_imag = np.linspace(-1.0, -1.0, size)
arr2_real = np.linspace(1.0, 1.0, size)
arr2_imag = np.linspace(-1.0, -1.0, size)
device_cnt = 1
level = 3

arr1_real_backup = arr1_real.copy()
arr1_imag_backup = arr1_imag.copy()
arr2_real_backup = arr2_real.copy()
arr2_imag_backup = arr2_imag.copy()

arr_cpu_real = np.zeros(size)
arr_cpu_imag = np.zeros(size)

runs = 10

t0 = time.time()

ptr1 = smartfp.smartfp_arr_create(arr1_real, arr1_imag, level, device_cnt)
ptr2 = smartfp.smartfp_arr_create(arr2_real, arr2_imag, level, device_cnt)
for run_index in range(runs):
    smartfp.smartfp_arr_compute(ptr1, ptr2)
smartfp.smartfp_arr_fetch(ptr1, arr1_real, arr1_imag, level, device_cnt)
smartfp.smartfp_arr_free(ptr1)
smartfp.smartfp_arr_free(ptr2)
    
print("gpu time: " + str(time.time()-t0))


t0 = time.time()
for _ in range(runs):
    for i in range(size):
        arr_cpu_real[i] = arr1_real_backup[i]*arr2_real_backup[i] - arr1_imag_backup[i]*arr2_imag_backup[i]
        arr_cpu_imag[i] = arr1_real_backup[i]*arr2_imag_backup[i] + arr1_imag_backup[i]*arr2_real_backup[i]
    arr1_real_backup = arr_cpu_real.copy()
    arr1_imag_backup = arr_cpu_imag.copy()
    # for i in range(10):
    #     print(str(arr_cpu_real[i]) + " + " + str(arr_cpu_imag[i]) + "i")
print("cpu time: " + str(time.time()-t0))

# print 10 items
# print("gpu results:")
# for i in range(10):
#     print(str(arr1_real[i]) + " + " + str(arr1_imag[i]) + "i")
# print("cpu results:")
# for i in range(10):
#     print(str(arr1_real_backup[i]) + " + " + str(arr1_imag_backup[i]) + "i")

for i in range(size):
    if abs(arr1_real[i] - arr1_real_backup[i]) > 0.0001:
        print("real error at index " + str(i) + ": " + str(arr1_real[i]) + " != " + str(arr1_real_backup[i]))
    if abs(arr1_imag[i] - arr1_imag_backup[i]) > 0.0001:
        print("imag error at index " + str(i) + ": " + str(arr1_imag[i]) + " != " + str(arr1_imag_backup[i]))

