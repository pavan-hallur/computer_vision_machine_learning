#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt

from depth_utils import *
from utils import *



#
# Task 2
#
# 3D reconstruction - via depth estimation - from stereo disparity.

img1 = np.float32(cv2.imread('img/aloe1.png')) / 255
img2 = np.float32(cv2.imread('img/aloe2.png')) / 255

# Define search range.
search_left = 80
search_right = 20

h, w, _ = img1.shape

disparity = np.zeros(img1.shape[:2], np.float32)

# TODO: Calculate the `disparity` (horizontal distance) for each pixel of `img1` to the corresponding (best matching) pixel in `img2`.
#  For a pixel (x, y), use plain pixel color as simple feature descriptor and search for the closest match in `img2` within the range [x - search_left, x + search_right].
#  Disparity should be relative to the image width (`w`) and can be positive or negative, depending on whether the shift is to the left or right.

# Compute summed color for each pixel in both images.
img1_summed_colors = np.sum(img1, axis=2)
img2_summed_colors = np.sum(img2, axis=2)

for row in range(h):
    # Current scanline in img1 and img2 with summed color.
    scanline1 = img1_summed_colors[row,:]
    scanline2 = img2_summed_colors[row,:]
    for col in range(w):
        # Summed color of current pixel in img1.
        color1 = scanline1[col]
        # Compute color distance to all pixels on scanline in img2.
        distance = color1 - scanline2
        # Take pixel with minimum distance
        disparity[row,col] = np.min(distance)

# # Go through all scanlines.
# for row in range(0,h):
#     # Create MxM matrix for current scanline. (Distance from x to matrix)
#     pixel_disparity = np.zeros((w, w))
#     for col in range(0,w):
#         # Summed color of current pixel on scanline.
#         img1_color = img1[row,col,0] + img1[row,col,1] + img1[row,col,2]
#         # Compare pixel with all pixels on that scanline
#         for i in range(0,w):
#             img2_color = img2[row,i,0] + img2[row,i,1] + img2[row,i,2]
#             distance = img1_color - img2_color
#             pixel_disparity[i,col] = distance
#     # Find best match.
#     for i in range(0,w):
#         disparity[row,i] = np.min(pixel_disparity[:,i])

# TODO: Convert to relative depth values from the `disparity`.
#  Pixels "closer to the camera" should be darker, pixels farther away should be brighter.
depth = disparity / w # Normalize by image width.


# Visualization
h_small = h // 4
w_small = w // 4
depth_small = cv2.medianBlur(depth, 5)
depth_small = cv2.resize(depth_small, dsize=(w_small, h_small), interpolation=cv2.INTER_AREA)

plt.figure(figsize=(14, 4))
showImages([("img1", img1), ("img2", img2), ("depth from disparity", depth_small)], 5, show_window_now=False, padding=[.01, .01, .01, .1])

plot3D(depth_small, (2, 5), (0, 3), (2, 2))
plt.show()
