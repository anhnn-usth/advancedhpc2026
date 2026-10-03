import time
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def _compute_local_histogram(chunk):

    lhisto = np.zeros(256, dtype=np.int64)
    unique, counts = np.unique(chunk, return_counts=True)
    lhisto[unique] = counts
    return lhisto


def compute_histogram_parallel(gray_img, num_workers):
    if num_workers is None:
        num_workers = cpu_count()

    chunks = np.array_split(gray_img, num_workers, axis=0)

    with Pool(processes=num_workers) as pool:
        local_histograms = pool.map(_compute_local_histogram, chunks)

    global_histo = np.sum(local_histograms, axis=0, dtype=np.int64)
    return global_histo


def compute_histogram_serial(gray_img):
    height, width = gray_img.shape
    histo = np.zeros(256, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            histo[gray_img[y, x]] += 1
    return histo

def equalize_histogram(gray_img, histo):
    height, width = gray_img.shape
    n = height * width

    p = histo.astype(np.float64) / float(n)

    c = np.cumsum(p)

    h = np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)

    equalized_img = h[gray_img]

    return equalized_img, c, h

if __name__ == "__main__":
    input_path = "input.jpg"
    output_path = "equalized_output.jpg"

    raw_img = Image.open(input_path).convert("L")
    gray_array = np.array(raw_img, dtype=np.uint8)
    h, w = gray_array.shape

    t0 = time.time()
    serial_histo = compute_histogram_serial(gray_array)
    t_serial = time.time() - t0

    t1 = time.time()
    parallel_histo = compute_histogram_parallel(gray_array)
    t_parallel = time.time() - t1

    speedup = t_serial / t_parallel if t_parallel > 0 else 1.0

    print(f"\n--- Labwork 9a Benchmark ---")
    print(f"Serial Time:       {t_serial * 1000.0:.2f} ms")
    print(f"Parallel Time:     {t_parallel * 1000.0:.2f} ms")
    print(f"Calculated Speedup: {speedup:.2f}x")

    equalized_array, cdf, lookup_table = equalize_histogram(gray_array, parallel_histo)

    Image.fromarray(equalized_array).save(output_path)
    print(f"\nEqualized image saved to: {output_path}")