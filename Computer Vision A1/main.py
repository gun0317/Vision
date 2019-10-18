import cv2
import numpy as np
import time
import A1_image_filtering as image_filtering
from my_func import *

lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
filename1 = "shapes"
filename2 = "lenna"

# Part.1
printing_gaussian_filter()
# shapes
res1 = save_nine_gaussian_2d(shapes, filename1)
res2 = comparing_gaussians(shapes, filename1, 5, 3)
# show image
my_img_resized = cv2.resize(res1, (shapes.shape[1] * 3 // 2, shapes.shape[0] * 3 // 2))
cv2.imshow("Nine Gaussians", my_img_resized / 255)
cv2.imshow("Difference Map", res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# lenna
res1 = save_nine_gaussian_2d(lenna, filename2)
res2 = comparing_gaussians(lenna, filename2, 5, 3)
# show image
my_img_resized = cv2.resize(res1, (lenna.shape[1] * 3 // 2, lenna.shape[0] * 3 // 2))
cv2.imshow("Nine Gaussians", my_img_resized / 255)
cv2.imshow("Difference Map", res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part.2

# Gaussian filtering
g2 = image_filtering.get_gaussian_filter_2d(7, 1.5)
shapes_g = image_filtering.cross_correlation_2d(shapes, g2)
lenna_g = image_filtering.cross_correlation_2d(lenna, g2)

# shapes
mag = show_compute_image_gradient(shapes_g, filename1)
suppressed = show_non_maximum_suppression_dir(shapes_g, filename1)
# show image
cv2.imshow("shapes_mag", mag / 255)
cv2.imshow("shapes_mag", suppressed / 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# lenna
mag = show_compute_image_gradient(lenna_g, filename2)
suppressed = show_non_maximum_suppression_dir(lenna_g, filename2)
# show image
cv2.imshow("shapes_mag", mag / 255)
cv2.imshow("shapes_mag", suppressed / 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part.3

# shapes
shapes_r = show_corner_response(shapes_g, filename1)
to_show = show_NMS(shapes_g, shapes_r, filename1)
img_rgb = show_suppressed_R(shapes, shapes_r, filename1)
# show image
cv2.imshow('response_raw', shapes_r)
cv2.imshow('greened', to_show / 255)
cv2.imshow('green_circled', img_rgb / 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# lenna
lenna_r = show_corner_response(lenna_g, filename2)
to_show = show_NMS(lenna_g, lenna_r, filename2)
img_rgb = show_suppressed_R(lenna, lenna_r, filename2)
# show image
cv2.imshow('response_raw', lenna_r)
cv2.imshow('greened', to_show / 255)
cv2.imshow('green_circled', img_rgb / 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
