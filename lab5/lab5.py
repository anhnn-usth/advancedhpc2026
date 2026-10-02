import time
from numba import cuda, float32, jit
import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True) #without this decorator, the runtime will extremely slow (up to 100x slower - tested)
def gaussian_blur_cpu(src, dst, filter_kernel):
    height, width, channels = src.shape
    
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                pixel_val = 0.0
                for j in range(7):
                    for i in range(7):
                        offset_x = x + i - 3
                        offset_y = y + j - 3
                        
                        if 0 <= offset_x < width and 0 <= offset_y < height:
                            pixel_val += src[offset_y, offset_x, c] * filter_kernel[j, i]
                
                dst[y, x, c] = pixel_val

@cuda.jit
def gaussian_blur_basic(src, dst, filter_kernel):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    height, width, channels = src.shape
    
    if x < width and y < height:
        for c in range(channels):
            pixel_val = 0.0
            for j in range(7):
                for i in range(7):
                    offset_x = x + i - 3
                    offset_y = y + j - 3
                    
                    if 0 <= offset_x < width and 0 <= offset_y < height:
                        pixel_val += src[offset_y, offset_x, c] * filter_kernel[j, i]
            
            dst[y, x, c] = pixel_val

@cuda.jit
def gaussian_blur_shared(src, dst, filter_kernel):
    s_filter = cuda.shared.array(shape=(7, 7), dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = tx + cuda.blockIdx.x * cuda.blockDim.x
    y = ty + cuda.blockIdx.y * cuda.blockDim.y
    height, width, channels = src.shape
    
    if ty < 7 and tx < 7:
        s_filter[ty, tx] = filter_kernel[ty, tx]
        
    cuda.syncthreads()
            
    if x < width and y < height:
        for c in range(channels):
            pixel_val = 0.0
            for j in range(7):
                for i in range(7):
                    offset_x = x + i - 3
                    offset_y = y + j - 3
                    
                    if 0 <= offset_x < width and 0 <= offset_y < height:
                        pixel_val += src[offset_y, offset_x, c] * s_filter[j, i]
            
            dst[y, x, c] = pixel_val

def main():
    image_path = 'input.jpg'
    img = plt.imread(image_path).astype(np.float32) / 255.0
        
    height, width, channels = img.shape

    host_kernel = np.array([
        [0,  0,  1,   2,   1,  0, 0],
        [0,  3, 13,  22,  13,  3, 0],
        [1, 13, 59,  97,  59, 13, 1],
        [2, 22, 97, 159,  97, 22, 2],
        [1, 13, 59,  97,  59, 13, 1],
        [0,  3, 13,  22,  13,  3, 0],
        [0,  0,  1,   2,   1,  0, 0]
    ], dtype=np.float32) / 1003.0

    dev_kernel = cuda.to_device(host_kernel)

    block_sizes = [(8, 8), (16, 8), (8, 16), (16, 32), (32, 16), (16, 16), (32, 32)]
    block_labels = [f"{b[0]}x{b[1]}" for b in block_sizes]
    
    cpu_times = []
    basic_times = []
    shared_times = []

    devInput = cuda.to_device(img)
    devOutput = cuda.device_array_like(devInput)

    print("--- Running CPU Gaussian Blur ---")
    for blockSize in block_sizes:
        start_cpu = time.time()
        
        cpu_img = np.zeros_like(img) 
        gaussian_blur_cpu(img, cpu_img, host_kernel)
        
        end_cpu = time.time()
        
        exec_time = end_cpu - start_cpu
        cpu_times.append(exec_time)
        print(f"CPU Baseline | Block/Tile Equivalent {blockSize} | Time: {exec_time:.6f} s")
        
    plt.imsave('output_cpu.jpg', np.clip(cpu_img, 0, 1))

    print("\n--- Running Basic Gaussian Blur (Without Shared Memory) ---")
    for blockSize in block_sizes:
        bx, by = blockSize
        gridSize = (int(np.ceil(width / bx)), int(np.ceil(height / by)))
        
        start_gpu = time.time()
        gaussian_blur_basic[gridSize, blockSize](devInput, devOutput, dev_kernel)
        cuda.synchronize()
        end_gpu = time.time()
        
        exec_time = end_gpu - start_gpu
        basic_times.append(exec_time)
        print(f"Global Mem | Block Size {blockSize} | Time: {exec_time:.6f} s")
        
    hostOutput = devOutput.copy_to_host()
    plt.imsave('output_basic.jpg', np.clip(hostOutput, 0, 1))

    print("\n--- Running Shared Memory Gaussian Blur ---")
    for blockSize in block_sizes:
        bx, by = blockSize
        gridSize = (int(np.ceil(width / bx)), int(np.ceil(height / by)))
        
        devOutput = cuda.device_array_like(devInput)
        
        start_gpu = time.time()
        gaussian_blur_shared[gridSize, blockSize](devInput, devOutput, dev_kernel)
        cuda.synchronize()
        end_gpu = time.time()
        
        exec_time = end_gpu - start_gpu
        shared_times.append(exec_time)
        print(f"Shared Mem | Block Size {blockSize} | Time: {exec_time:.6f} s")

    hostOutput = devOutput.copy_to_host()
    plt.imsave('output_shared.jpg', np.clip(hostOutput, 0, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(block_labels, cpu_times, marker='o', linestyle='-', color='forestgreen', linewidth=2, markersize=8, label='CPU Loop')
    ax1.set_xlabel('Block/Tile Configuration')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('CPU Execution Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(block_labels, basic_times, marker='o', linestyle='-', color='firebrick', linewidth=2, markersize=8, label='Without Shared Memory')
    ax2.plot(block_labels, shared_times, marker='s', linestyle='-', color='royalblue', linewidth=2, markersize=8, label='With Shared Memory')
    ax2.set_xlabel('GPU Thread Block Size')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('GPU Execution Time vs Block Size')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    for i, txt in enumerate(basic_times):
        ax2.annotate(f'{txt:.4f}s', (block_labels[i], basic_times[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                     
    for i, txt in enumerate(shared_times):
        ax2.annotate(f'{txt:.4f}s', (block_labels[i], shared_times[i]), 
                     textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('blur_performance.jpg', dpi=300)

if __name__ == "__main__":
    main()