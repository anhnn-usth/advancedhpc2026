from numba import cuda

print("--- GPU Detection ---")
cuda.detect()

device = cuda.get_current_device()

print("\n--- GPU Information ---")
print(f"Device ID: {device.id}")
print(f"Device Name: {device.name.encode('utf-8')}")

print(f"Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")

free_mem, total_mem = cuda.current_context().get_memory_info()
print(f"Total Memory: {total_mem / (1024**3):.2f} GB")
print(f"Free Memory: {free_mem / (1024**3):.2f} GB")

cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128,
    (8,9) : 128,
    (9,0) : 128,
    (10,0) : 128,
    (12,0) : 128
    }
# the above dictionary should result in a value of "None" if a cc match 
# is not found.  The dictionary needs to be extended as new devices become
# available, and currently does not account for all Jetson devices
device = cuda.get_current_device()
my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
my_cc = device.compute_capability
cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
total_cores = cores_per_sm*my_sms
print("GPU compute capability: " , my_cc)
print("GPU total number of SMs: " , my_sms)
print("total cores: " , total_cores)

