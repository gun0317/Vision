from my_func import *
import math
import A1_image_filtering as image_filtering


# 2-2 *need to pass the gaussian-filtered image
def compute_image_gradient(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    x = image_filtering.cross_correlation_2d(img, sobel_x)
    y = image_filtering.cross_correlation_2d(img, sobel_y)

    mag = np.empty([img.shape[0], img.shape[1]])
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            mag[r][c] = math.sqrt(x[r][c] * x[r][c] + y[r][c] * y[r][c])

    theta = np.arctan2(y, x)

    return mag, theta


# In[744]:


# 2-3
def non_maximum_suppression_dir(mag, degree):
    maxR, maxC = mag.shape
    result = np.zeros([maxR, maxC], np.float32)
    degree += np.pi
    degree = degree / np.pi * 180

    for r in range(1, maxR - 1):
        for c in range(1, maxC - 1):
            try:
                ref1 = 0
                ref2 = 0

                if (0 <= degree[r][c] < 22.5) or (337.5 <= degree[r][c] <= 360) or (157.5 <= degree[r][c] <= 202.5):
                    ref1 = mag[r][c + 1]
                    ref2 = mag[r][c - 1]
                elif (22.5 <= degree[r][c] <= 67.5) or (202.5 <= degree[r][c] <= 247.5):
                    ref1 = mag[r + 1][c - 1]
                    ref2 = mag[r - 1][c + 1]
                elif (67.5 <= degree[r][c] <= 112.5) or (247.5 <= degree[r][c] <= 292.5):
                    ref1 = mag[r - 1][c]
                    ref2 = mag[r + 1][c]
                elif (112.5 <= degree[r][c] <= 157.5) or (292.5 <= degree[r][c] <= 337.5):
                    ref1 = mag[r - 1][c - 1]
                    ref2 = mag[r + 1][c + 1]

                if (mag[r][c] >= ref1) and (mag[r][c] >= ref2):
                    result[r][c] = mag[r][c]
                else:
                    result[r][c] = 0

            except IndexError as e:
                pass

    return result
