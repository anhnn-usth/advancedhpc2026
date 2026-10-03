import math
import numpy as np
from PIL import Image
from numba import cuda


@cuda.jit
def binarized_kernel(src, dst, width, height, tau):
    x, y = cuda.grid(2)
    if x < width and y < height:
        gray = (float(src[y, x, 0]) + float(src[y, x, 1]) + float(src[y, x, 2])) / 3.0

        if gray >= tau:
            dst[y, x] = 255
        else:
            dst[y, x] = 0


@cuda.jit
def brightness_kernel(d_img, d_out, width, height, channels, delta):
    x, y = cuda.grid(2)
    if x < width and y < height:
        for c in range(channels):
            val = int(d_img[y, x, c]) + delta
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            d_out[y, x, c] = val


@cuda.jit
def blend_kernel(d_img1, d_img2, d_out, width, height, channels, c):
    x, y = cuda.grid(2)
    if x < width and y < height:
        for ch in range(channels):
            v1 = float(d_img1[y, x, ch])
            v2 = float(d_img2[y, x, ch])
            blended = c * v1 + (1.0 - c) * v2

            if blended < 0.0:
                blended = 0.0
            elif blended > 255.0:
                blended = 255.0

            d_out[y, x, ch] = int(blended)



def binarize_image_gpu(img_array, tau, block_dim=(16, 16)):
    height, width, _ = img_array.shape

    d_img = cuda.to_device(img_array)
    d_out = cuda.device_array((height, width), dtype=np.uint8)

    grid_dim = (
        math.ceil(width / block_dim[0]),
        math.ceil(height / block_dim[1])
    )

    binarized_kernel[grid_dim, block_dim](d_img, d_out, width, height, tau)
    return d_out.copy_to_host()


def adjust_brightness_gpu(img_array, delta, block_dim=(16, 16)):
    height, width, channels = img_array.shape

    d_img = cuda.to_device(img_array)
    d_out = cuda.device_array_like(img_array)

    grid_dim = (
        math.ceil(width / block_dim[0]),
        math.ceil(height / block_dim[1])
    )

    brightness_kernel[grid_dim, block_dim](d_img, d_out, width, height, channels, delta)
    return d_out.copy_to_host()


def blend_images_gpu(img1_array, img2_array, c=0.5, block_dim=(16, 16)):
    if img1_array.shape != img2_array.shape:
        raise ValueError(f"Shape mismatch: {img1_array.shape} vs {img2_array.shape}")

    height, width, channels = img1_array.shape

    d_img1 = cuda.to_device(img1_array)
    d_img2 = cuda.to_device(img2_array)
    d_out = cuda.device_array_like(img1_array)

    grid_dim = (
        math.ceil(width / block_dim[0]),
        math.ceil(height / block_dim[1])
    )

    blend_kernel[grid_dim, block_dim](d_img1, d_img2, d_out, width, height, channels, c)
    return d_out.copy_to_host()


def binarize_image(img_array, tau=128):
    img_float = img_array.astype(np.float32)
    gray = (img_float[:, :, 0] + img_float[:, :, 1] + img_float[:, :, 2]) / 3.0

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray >= tau] = 255
    return binary


def adjust_brightness(img_array, delta):
    adjusted = img_array.astype(np.int32) + delta

    adjusted[adjusted < 0] = 0
    adjusted[adjusted > 255] = 255

    return adjusted.astype(np.uint8)


def blend_images(img1_array, img2_array, c = 0.5):
    blended = c * img1_array.astype(np.float32) + (1.0 - c) * img2_array.astype(np.float32)

    blended[blended < 0.0] = 0.0
    blended[blended > 255.0] = 255.0

    return blended.astype(np.uint8)

def main():
    img1 = np.array(Image.open("input.jpg"))
    img2 = np.array(Image.open("input2.jpg"))

    binarized_result = binarize_image(img1, tau=128)
    Image.fromarray(binarized_result).save("output_binarized.jpg")
    print(f"Binarized image saved to {"output_binarized.jpg"}")

    bright_result = adjust_brightness(img1, delta=50)
    Image.fromarray(bright_result).save("output_50.jpg")
    print(f"Brightness increased saved to {"output_50.jpg"}")

    dark_result = adjust_brightness(img1, delta=-50)
    Image.fromarray(dark_result).save("output_-50.jpg")
    print(f"Brightness decreased saved to {"output_-50.jpg"}")

    blended_result = blend_images(img1, img2, c=0.6)
    Image.fromarray(blended_result).save("output_blended.jpg")
    print(f"Blended image saved to {"output_blended.jpg"}")

    bin_gpu = binarize_image_gpu(img1, tau=128, block_dim=(16, 16))
    Image.fromarray(bin_gpu).save("output_binarized_gpu.jpg")
    print(f"GPU: Binarized image saved to {"output_binarized_gpu.jpg"}")

    bright_gpu = adjust_brightness_gpu(img1, delta=50, block_dim=(16, 16))
    Image.fromarray(bright_gpu).save("output_50_gpu.jpg")
    print(f"GPU: Brightness (+50) image saved to {"output_50_gpu.jpg"}")

    dark_gpu = adjust_brightness_gpu(img1, delta=-50, block_dim=(16, 16))
    Image.fromarray(dark_gpu).save("output_-50_gpu.jpg")
    print(f"GPU: Brightness (-50) image saved to {"output_-50_gpu.jpg"}")

    blend_gpu = blend_images_gpu(img1, img2, c=0.6, block_dim=(16, 16))
    Image.fromarray(blend_gpu).save("output_blended_gpu.jpg")
    print(f"GPU: Blended image saved to {"output_blended_gpu.jpg"}")


if __name__ == "__main__":
    main()