import numpy as np
from PIL import Image


def RGB2HSV(rgb_aos):

    height, width, _ = rgb_aos.shape

    H = np.zeros((height, width), dtype=np.float32)
    S = np.zeros((height, width), dtype=np.float32)
    V = np.zeros((height, width), dtype=np.float32)

    rgb_norm = rgb_aos.astype(np.float32) / 255.0
    r = rgb_norm[:, :, 0]
    g = rgb_norm[:, :, 1]
    b = rgb_norm[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    V[:] = c_max


    non_zero_max = c_max > 0.0
    S[non_zero_max] = delta[non_zero_max] / c_max[non_zero_max]


    nonzero_delta = delta > 0.0

    mask_r = nonzero_delta & (c_max == r)
    mask_g = nonzero_delta & (c_max == g) & (~mask_r)
    mask_b = nonzero_delta & (c_max == b) & (~mask_r) & (~mask_g)

    H[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)

    H[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)

    H[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    H = (H % 360.0 + 360.0) % 360.0

    return H, S, V


def HSV2RGB(H, S, V):
    height, width = H.shape
    rgb_aos = np.zeros((height, width, 3), dtype=np.uint8)

    d = H / 60.0
    hi = np.floor(d).astype(np.int32) % 6
    f = d - np.floor(d)

    l = V * (1.0 - S)
    m = V * (1.0 - f * S)
    n = V * (1.0 - (1.0 - f) * S)

    r = np.zeros_like(V)
    g = np.zeros_like(V)
    b = np.zeros_like(V)

    c0 = hi == 0
    r[c0], g[c0], b[c0] = V[c0], n[c0], l[c0]

    c1 = hi == 1
    r[c1], g[c1], b[c1] = m[c1], V[c1], l[c1]

    c2 = hi == 2
    r[c2], g[c2], b[c2] = l[c2], V[c2], n[c2]

    c3 = hi == 3
    r[c3], g[c3], b[c3] = l[c3], m[c3], V[c3]

    c4 = hi == 4
    r[c4], g[c4], b[c4] = n[c4], l[c4], V[c4]

    c5 = hi == 5
    r[c5], g[c5], b[c5] = V[c5], l[c5], m[c5]

    rgb_aos[:, :, 0] = np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)
    rgb_aos[:, :, 1] = np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8)
    rgb_aos[:, :, 2] = np.clip(np.round(b * 255.0), 0, 255).astype(np.uint8)

    return rgb_aos


if __name__ == "__main__":
    input_path = "input2.jpg"
    output_path = "reconstructed.jpg"

    input_img = Image.open(input_path)
    original_aos = np.array(input_img)

    H, S, V = RGB2HSV(original_aos)

    reconstructed_aos = HSV2RGB(H, S, V)

    Image.fromarray(reconstructed_aos).save(output_path)

    abs_diff = np.abs(original_aos.astype(np.int32) - reconstructed_aos.astype(np.int32))
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)

    print(f"Max Pixel Difference:  {max_err}")
    print(f"Mean Pixel Difference: {mean_err:.4f}")