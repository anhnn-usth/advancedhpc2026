import time
from turtle import width

from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
@cuda.jit
def grayscale(src, dst):
    # where are we in the input?
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

def main():
    image_path = 'input.jpg'
    img = plt.imread(image_path)

    # Flatten 1d image 
    img_flat = img.reshape(-1, 3)
    height, width, channels = img.shape

    cpu_gray_flat = np.zeros(width * height, dtype=np.float32)

    start_cpu = time.time()
    for i in range(width * height):
        r = img_flat[i, 0]
        g = img_flat[i, 1]
        b = img_flat[i, 2]
        #https://biology.stackexchange.com/questions/110562/what-is-the-best-estimate-of-rgb-color-sensitivity-of-human-eye
        cpu_gray_flat[i] = 0.64 * r + 0.32 * g + 0.02 * b
    end_cpu = time.time()

    cpu_time = end_cpu - start_cpu
    cpu_img = cpu_gray_flat.reshape((height, width))
    print(f"CPU Grayscale Time: {cpu_time:.6f} seconds")

    plt.imsave('cpu_output.jpg', cpu_img, cmap='gray')

    devInput = cuda.to_device(img_flat)

    # since devInput has flattened, use device_array_like
    devOutput = cuda.device_array_like(devInput)

    pixelCount = width * height
    blockSize = 64
    gridSize = pixelCount // blockSize
    start_gpu = time.time()
    grayscale[gridSize, blockSize](devInput, devOutput)

    end_gpu = time.time()
    print(f"GPU execution time: {end_gpu - start_gpu} seconds")

    hostOutput = devOutput.copy_to_host()

    result_img = hostOutput.reshape(height, width, channels)

    plt.imsave('output_gpu.jpg', result_img)

if __name__ == "__main__":
    main()

