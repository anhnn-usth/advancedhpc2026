import math
import time

from PIL import Image

from numba import cuda
import numpy as np

@cuda.jit
def rgb_to_gray_kernel(rgb_img, gray_img):
    x, y = cuda.grid(2)
    height, width, _ = rgb_img.shape
    if y < height and x < width:
        gray_img[y, x] = (rgb_img[y, x, 0] + rgb_img[y, x, 1] + rgb_img[y, x, 2]) / 3.0

@cuda.jit
def reduce_min_max_kernel(data, block_mins, block_maxs):
    s_min = cuda.shared.array(256, dtype=np.float32)
    s_max = cuda.shared.array(256, dtype=np.float32)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)
    h, w = data.shape

    row = gid // w
    col = gid % w
    val = data[row, col]
    s_min[tid] = val
    s_max[tid] = val

    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            if s_min[tid + stride] < s_min[tid]:
                s_min[tid] = s_min[tid + stride]
            if s_max[tid + stride] > s_max[tid]:
                s_max[tid] = s_max[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        block_mins[cuda.blockIdx.x] = s_min[0]
        block_maxs[cuda.blockIdx.x] = s_max[0]

def find_min_max_gpu(d_gray, block_dim_1d=256):
    n = d_gray.size
    grid_dim = math.ceil(n / block_dim_1d)

    d_block_mins = cuda.device_array(grid_dim, dtype=np.float32)
    d_block_maxs = cuda.device_array(grid_dim, dtype=np.float32)

    reduce_min_max_kernel[grid_dim, block_dim_1d](d_gray, d_block_mins, d_block_maxs)

    if grid_dim > 1:
        h_mins = d_block_mins.copy_to_host()
        h_maxs = d_block_maxs.copy_to_host()
        return float(np.min(h_mins)), float(np.max(h_maxs))
    else:
        return float(d_block_mins.copy_to_host()[0]), float(d_block_maxs.copy_to_host()[0])

@cuda.jit
def stretch_kernel(gray_in, stretched_out, min_val, max_val):
    x, y = cuda.grid(2)
    height, width = gray_in.shape

    if y < height and x < width:
        val = gray_in[y, x]
        scaled = ((val - min_val) / (max_val - min_val)) * 255.0
        stretched_out[y, x] = np.uint8(max(0.0, min(255.0, round(scaled))))

def run_grayscale_stretch(image_path, block_size_2d=(16, 16)):
    img = Image.open(image_path)
    h_rgb = np.array(img, dtype=np.float32)
    h, w, _ = h_rgb.shape

    d_rgb = cuda.to_device(h_rgb)
    d_gray = cuda.device_array((h, w), dtype=np.float32)
    d_out = cuda.device_array((h, w), dtype=np.uint8)

    threads_per_block_2d = block_size_2d
    blocks_x = math.ceil(w / threads_per_block_2d[0])
    blocks_y = math.ceil(h / threads_per_block_2d[1])
    grid_2d = (blocks_x, blocks_y)

    cuda.synchronize()
    start_time = time.time()

    rgb_to_gray_kernel[grid_2d, threads_per_block_2d](d_rgb, d_gray)

    min_val, max_val = find_min_max_gpu(d_gray, block_dim_1d=256)

    stretch_kernel[grid_2d, threads_per_block_2d](d_gray, d_out, min_val, max_val)

    cuda.synchronize()
    gpu_time = (time.time() - start_time) * 1000

    h_out = d_out.copy_to_host()
    stretched_img = Image.fromarray(h_out)
    stretched_img.save("stretched_result.png")

    print(f"Block Size: {block_size_2d} | Range: [{min_val:.2f}, {max_val:.2f}] | GPU Execution Time: {gpu_time:.2f} ms")
    return gpu_time


if __name__ == "__main__":
    input_image_path = "input2.jpg"

    block_sizes = [(8, 8), (16, 16), (32, 16), (32, 32)]
    print(f"Benchmarking different 2D block sizes for {input_image_path}:")
    for bs in block_sizes:
        run_grayscale_stretch(input_image_path, block_size_2d=bs)