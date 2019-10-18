import math
import numpy as np
from my_func import *


# 1-1 (a)
def cross_correlation_1d(img, kernel):
    k = (kernel.shape[0] // 2)
    if k == 0:
        k = kernel.shape[1] // 2
    img_row = img.shape[0]
    img_col = img.shape[1]

    img = pad_1d(img, kernel)
    correlation = np.zeros([img_row, img_col], np.float32)

    # row vector
    if kernel.shape[0] == 1:
        for r in range(0, img_row):
            for c in range(k, k + img_col):
                ref_matrix = img[r:r + 1, c - k:c + k + 1]
                correlation[r][c - k] = np.sum(np.multiply(kernel, ref_matrix))
    else:
        for r in range(k, k + img_row):
            for c in range(0, img_col):
                ref_matrix = img[r - k:r + k + 1, c:c + 1]
                correlation[r - k][c] = np.sum(np.multiply(kernel, ref_matrix))

    return correlation


# 1-1 (a)
def cross_correlation_2d(img, kernel):
    k = (kernel.shape[0] // 2)
    img_row = img.shape[0]
    img_col = img.shape[1]

    img = pad_2d(img, kernel)
    correlation = np.zeros([img_row, img_col], np.float32)

    for r in range(k, k + img_row):
        for c in range(k, k + img_col):
            ref_matrix = img[r - k:r + k + 1, c - k:c + k + 1]
            correlation[r - k][c - k] = np.sum(np.multiply(kernel, ref_matrix))

    return correlation


# 1-2 (a)
def get_gaussian_filter_1d(size, sigma):
    kernel = np.zeros([1, size], np.float32)
    s = size // 2
    for x in range(-s, s + 1):
        kernel[0][x + s] = pow(math.e, -1 * (x * x / (2 * sigma * sigma))) / (math.sqrt(2 * math.pi) * sigma)

    # Normalization
    kernel *= 1 / kernel.sum()

    return kernel


# 1-2 (a)
def get_gaussian_filter_2d(size, sigma):
    kernel = np.zeros([size, size], np.float32)
    s = size // 2

    for r in range(-s, s + 1):
        for c in range(-s, s + 1):
            kernel[r + s][c + s] = pow(math.e, -1 * ((r * r + c * c) / (2 * sigma * sigma))) / (
                    2 * math.pi * sigma * sigma)

    # Normalization
    kernel *= 1 / kernel.sum()

    return kernel
