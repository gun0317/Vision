import numpy as np
import cv2
import A1_image_filtering as image_filtering
import A1_edge_detection as edge_detection
import A1_corner_detection as corner_detection
import time


def copy_corner(img, r1, r2, c1, c2, refR, refC):
    img[r1:r2][c1:c2] = img[refR][refC]


def copy_bar(img, r1, r2, c1, c2, refR):  # horizontal
    for r in range(r1, r2):
        for c in range(c1, c2):
            img[r][c] = img[refR][c]


def copy_post(img, r1, r2, c1, c2, refC):  # vertical
    for r in range(r1, r2):
        for c in range(c1, c2):
            img[r][c] = img[r][refC]


def pad_1d(img, kernel):
    k = kernel.shape[0] // 2
    if k == 0:
        k = kernel.shape[1] // 2

    img_row = img.shape[0]
    img_col = img.shape[1]
    max_row_img = k + img_row - 1
    max_col_img = k + img_col - 1

    # row vector
    if kernel.shape[0] == 1:
        # Padding black pixels
        padding_post = np.zeros([img_row, k])
        img = np.hstack([padding_post, img])
        img = np.hstack([img, padding_post])
        # Post
        copy_post(img, 0, img_row, 0, k, k)
        copy_post(img, 0, img_row, max_col_img, k + max_col_img + 1, max_col_img)
    # column vector
    else:
        # Padding black pixels
        padding_bar = np.zeros([k, img_col])
        img = np.vstack([padding_bar, img])
        img = np.vstack([img, padding_bar])
        # Bar
        copy_bar(img, 0, k, 0, img_col, k)
        copy_bar(img, max_row_img, max_row_img + k, 0, img_col, max_row_img)

    return img


def pad_2d(img, kernel):
    k = kernel.shape[0] // 2
    img_row = img.shape[0]
    img_col = img.shape[1]
    max_row = 2 * k + img_row
    max_col = 2 * k + img_col
    max_row_img = k + img_row - 1
    max_col_img = k + img_col - 1

    # Padding black pixels
    padding_bar = np.zeros([k, img_col], np.float32)
    padding_post = np.zeros([img_row + 2 * k, k], np.float32)
    img = np.vstack([padding_bar, img])
    img = np.vstack([img, padding_bar])
    img = np.hstack([padding_post, img])
    img = np.hstack([img, padding_post])

    # Copying the color of original img
    # Corner
    copy_corner(img, 0, k, 0, k, k, k)
    copy_corner(img, 0, k, k + img_col, max_col, k, max_col_img)
    copy_corner(img, img_row + k, max_row, 0, k, max_row_img, k)
    copy_corner(img, max_row_img + 1, max_row, max_row_img + 1, max_row, max_row_img, max_col_img)
    # Bar
    copy_bar(img, 0, k, k, k + img_col, k)
    copy_bar(img, max_col_img, max_row, k, k + img_col, max_row_img)
    # Post
    copy_post(img, k, max_row_img + 1, 0, k, k)
    copy_post(img, k, max_row_img + 1, max_col_img + 1, max_col, max_col_img)

    return img


# 1-2 (c)
def printing_gaussian_filter():
    g1 = image_filtering.get_gaussian_filter_1d(5, 1)
    g2 = image_filtering.get_gaussian_filter_2d(5, 1)
    print("1D Gaussian filter (5, 1)")
    print(g1)
    print("2D Gaussian filter (5, 1)")
    print(g2)


# 1-2 (d) *need to used for both lenna and shapes
def save_nine_gaussian_2d(img, filename):
    k_size = [5, 11, 17]
    sigmas = [1, 6, 11]
    my_img = np.empty([0, img.shape[1] * 3], np.float32)

    for k in k_size:
        local_img = np.empty([img.shape[0], 0])
        for s in sigmas:
            g = image_filtering.get_gaussian_filter_2d(k, s)
            res = image_filtering.cross_correlation_2d(img, g)
            # Putting text
            txt = str(k) + 'x' + str(k) + ' s=' + str(s)
            cv2.putText(res, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0)
            local_img = np.hstack([local_img, res])
        my_img = np.vstack([my_img, local_img])

    cv2.imwrite('./result/part_1_gaussian_filtered_' + filename + '.png', my_img)

    return my_img


# 1-2 (e) *need to used for both lenna and shape
def comparing_gaussians(img, filename, kernel_size, sigma):
    # 1d kernel
    g1 = image_filtering.get_gaussian_filter_1d(kernel_size, sigma)
    g1_T = g1.T
    s = time.time()
    img_1d = image_filtering.cross_correlation_1d(img, g1_T)
    img_1d = image_filtering.cross_correlation_1d(img_1d, g1)
    print("Time taken(1d Gaussian filters) for " + filename + ': ', time.time() - s)
    # 2d kernel
    s = time.time()
    g2 = image_filtering.get_gaussian_filter_2d(kernel_size, sigma)
    img_2d = image_filtering.cross_correlation_2d(img, g2)
    print("Time taken(2d Gaussian filters) for " + filename + ': ', time.time() - s)

    diff = abs(abs(img_1d) - abs(img_2d))
    diff /= 255

    cv2.imwrite('./result/part_1_gaussian_difference_map_' + filename + '.png', diff)
    print("Sum of intensity difference: ", np.sum(diff))

    return diff


# 2-2
def show_compute_image_gradient(img1, filename):
    start = time.time()
    mag1, theta1 = edge_detection.compute_image_gradient(img1)
    end = time.time()
    print("Time taken(compute_gradient) for " + filename + ": ", end - start)
    cv2.imwrite('./result/part_2_edge_raw_' + filename + '.png', mag1)
    return mag1

# 2-3
def show_non_maximum_suppression_dir(img1, filename):
    # Getting magnitude, gradient
    mag1, theta1 = edge_detection.compute_image_gradient(img1)

    start = time.time()
    suppressed1 = edge_detection.non_maximum_suppression_dir(mag1, theta1)
    end = time.time() - start
    print("Time taken(suppression) for " + filename + ": ", end)
    cv2.imwrite('./result/part_2_edge_sup_' + filename + '.png', suppressed1)
    return suppressed1


# 3-2
def show_corner_response(img, filename):
    s = time.time()
    r = corner_detection.compute_corner_response(img)
    print("Time taken(corner detecting_raw) for " + filename + ": ", time.time() - s)
    cv2.imwrite('./result/part_3_corner_raw_' + filename + '.png', r * 255)
    return r


# 3-3
def show_NMS(img, r, filename):
    response = r.copy()
    to_show = grayscale_to_rgb(img).copy()
    maxR = to_show.shape[0]
    maxC = to_show.shape[1]
    for r in range(maxR):
        for c in range(maxC):
            if response[r][c] >= 0.1:
                to_show[r][c][0] = 0
                to_show[r][c][1] = 255
                to_show[r][c][2] = 0
    cv2.imwrite('./result/part_3_corner_bin_' + filename + '.png', to_show)
    return to_show


def grayscale_to_rgb(g):
    # adding axis
    g = g[:, :, np.newaxis]
    g = np.append(g, np.zeros([g.shape[0], g.shape[1], 2]), axis=2)
    maxR = g.shape[0]
    maxC = g.shape[1]
    for r in range(maxR):
        for c in range(maxC):
            to_copy = g[r][c][0]
            g[r][c][1] = g[r][c][2] = to_copy

    return g


def show_suppressed_R(img, R, filename):
    s = time.time()
    suppressed_R = corner_detection.non_maximum_suppression_win(R, 11)
    print("Time taken(non_max_suppression) for " + filename + ": ", time.time() - s)
    img_rgb = grayscale_to_rgb(img)
    maxR = img_rgb.shape[0]
    maxC = img_rgb.shape[1]
    for r in range(maxR):
        for c in range(maxC):
            if suppressed_R[r][c] >= 0.1:
                cv2.circle(img_rgb, (c, r), 4, (0, 255, 0), 1)
    cv2.imwrite('./result/part_3_corner_sup_' + filename + '.png', img_rgb)
    return img_rgb

