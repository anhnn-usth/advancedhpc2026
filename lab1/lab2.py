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

import os
# Check general GPU specifications via the system CLI
os.system("nvidia-smi")
