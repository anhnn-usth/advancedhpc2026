import numpy as np
from PIL import Image

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


if __name__ == "__main__":
    main()