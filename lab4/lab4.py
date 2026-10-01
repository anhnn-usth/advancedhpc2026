import time
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    height, width, _ = src.shape
    if x < width and y < height:
        g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) / 3)
        dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = g

def main():
    image_path = 'input.jpg'
    img = plt.imread(image_path).astype(np.float32)
    height, width, channels = img.shape

    block_sizes = [(8, 8), (16,8),(8,16),(16,32),(32,16), (16, 16), (32, 32)]
    block_labels = [f"{b[0]}x{b[1]}" for b in block_sizes]
    
    cpu_times = []
    gpu_times = []

    devInput = cuda.to_device(img)
    devOutput = cuda.device_array_like(devInput)

    for blockSize in block_sizes:
        bx, by = blockSize

        start_cpu = time.time()
        cpu_img = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                cpu_img[y, x] = (img[y, x, 0] + img[y, x, 1] + img[y, x, 2]) / 3
                        
        end_cpu = time.time()
        cpu_time = end_cpu - start_cpu
        cpu_times.append(cpu_time)
        print(f"CPU Tile Size {blockSize} | Time: {cpu_time:.6f} s")

    plt.imsave('cpu_output.jpg', cpu_img, cmap='gray')

    for blockSize in block_sizes:
        bx, by = blockSize
        gridSize = (width // bx, height // by)
        
        start_gpu = time.time()
        
        grayscale[gridSize, blockSize](devInput, devOutput)

        end_gpu = time.time()
        gpu_time = end_gpu - start_gpu
        gpu_times.append(gpu_time)
        
        print(f"GPU Block Size {blockSize} | Time: {gpu_time:.6f} s")

    hostOutput = devOutput.copy_to_host() / 255.0
    result_img = hostOutput.reshape(height, width, channels)
    plt.imsave('output_gpu.jpg', result_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(block_labels, cpu_times, marker='o', linestyle='-', color='firebrick', linewidth=2, markersize=8)
    ax1.set_xlabel('CPU Tile/Block Size')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('CPU Execution Time by Tile Size')
    ax1.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(cpu_times):
        ax1.annotate(f'{txt:.4f}s', (block_labels[i], cpu_times[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    ax2.plot(block_labels, gpu_times, marker='s', linestyle='-', color='royalblue', linewidth=2, markersize=8)
    ax2.set_xlabel('GPU Thread Block Size')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('GPU Execution Time by Block Size')
    ax2.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(gpu_times):
        ax2.annotate(f'{txt:.5f}s', (block_labels[i], gpu_times[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('per.png', dpi=300)

if __name__ == "__main__":
    main()