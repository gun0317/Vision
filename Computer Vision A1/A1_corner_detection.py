from my_func import *
import math
import A1_image_filtering as image_filtering


def compute_corner_response(img):
    response = np.zeros([img.shape[0], img.shape[1]], np.float32)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = image_filtering.cross_correlation_2d(img, sobel_x)
    Iy = image_filtering.cross_correlation_2d(img, sobel_y)
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2
    h, w = img.shape
    k = 0.04
    w_size = 5
    off = w_size // 2
    for y in range(off, h - off):
        for x in range(off, w - off):
            xx = np.sum(Ixx[y - off:y + 1 + off, x - off:x + 1 + off])
            yy = np.sum(Iyy[y - off:y + 1 + off, x - off:x + 1 + off])
            xy = np.sum(Ixy[y - off:y + 1 + off, x - off:x + 1 + off])

            # Find determinant and trace, use to get corner response
            A = (xx * yy) - (xy ** 2)
            B = xx + yy
            r = A - k * (B ** 2)

            response[y][x] = r

    response = np.array(response)
    response[response < 0] = 0

    response /= np.max(response)

    return response


def non_maximum_suppression_win(R, winSize):
    threshold = 0.1
    off = winSize//2
    h, w = R.shape
    for y in range(off, h-off):
        for x in range(off, w-off):
            if R[y][x] != 0:
                localMax = 0
                for i in range(0,off):
                    refMatrix = R[y-off+i][x-off:x+off+1]
                    localMax = np.max(refMatrix)
                if R[y][x] < localMax or R[y][x] < threshold:
                    R[y][x] = 0

    return R