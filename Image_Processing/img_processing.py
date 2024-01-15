#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 4
#
"""
Image handling:
1. Load image.
2. Show image with showImage.
3. Complete and print image stats.
4. Switch red and blue image channels.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def showImage(img, show_window_now = True):
    # Convert the channel order of an image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt_img = plt.imshow(img_rgb)
    if show_window_now:
        plt.show()
    return plt_img

# Load the image "img/hummingbird_from_pixabay.png" with OpenCV (`cv2`) to the variable `img` and show it with `showImage(img)`.
img = cv2.imread("img/hummingbird_from_pixabay.png")

# Show the image.
showImage(img)

def imageStats(img):
    print("Image stats:")
    print("Shape :", img.shape)
    print("Width :", img.shape[0])
    print("Height:", img.shape[1])
    print("Number of channels:", img.shape[2])

# Print image stats of the hummingbird image.
imageStats(img)

# Change the color of the hummingbird to blue by swapping red and blue image channels.
img_rgb = img[:,:,::-1]

# Store the modified image as "blue_hummingbird.png" to your hard drive.
cv2.imwrite("img/blue_hummingbird.png", img_rgb)

#
# Task 5
#
"""
Image handling:
1. Convert img to HSV.
2. Change hue channel by adding a dynamic hue_offset
3. Show modified image.
"""

from matplotlib.widgets import Slider

# Prepare to show the original image and keep a reference so that we can update the image plot later.
plt.figure(figsize=(4, 6))
plt_img = showImage(img, False)

# Convert the original image to HSV color space.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def img_update(hue_offset):
    print("Set hue offset to " + str(hue_offset))
    # Change the hue channel of the HSV image by `hue_offset`.
    # Mind that hue values in OpenCV range from 0-179.
    mask = cv2.inRange(img_hsv[:,:,0], 3, 29)
    cv2.imshow("Mask", mask)
    cv2.waitKey(1)

    temp        = cv2.copyTo(img_hsv, mask=None)
    # Change whole image.
    #temp[:,:,0] = (temp[:,:,0]+hue_offset)%180
    # Change only the bird.
    temp[:,:,0] = np.where(mask != 0, (temp[:,:,0]+hue_offset)%180, temp[:,:,0])

    # Convert the modified HSV image back to RGB
    # and update the image in the plot window using `plt_img.set_data(img_rgb)`.
    img_rgb = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
    plt_img.set_data(img_rgb)

# Create an interactive slider for the hue value offset.
ax_hue = plt.axes([0.1, 0.04, 0.8, 0.06]) # x, y, width, height
slider_hue = Slider(ax=ax_hue, label='Hue', valmin=0, valmax=180, valinit=0, valstep=1)
slider_hue.on_changed(img_update)

# Now actually show the plot window
plt.show()
