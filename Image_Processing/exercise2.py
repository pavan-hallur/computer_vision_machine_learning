#!/usr/bin/env python3

####################
####  Task 2.1  ####
####################
"""
1. Create a list with values 1–20.
2. Use a list comprehension to square all odd values in this list.
3. Request 4 numbers and sort the numbers in ascending order.
"""
# my_list = list(range(1,21))
# print(my_list)

# my_list = [x**2 if x%2 != 0 else x for x in my_list]
# print(my_list)

# input_list = list()
# for i in range(4):
#     input_list.append(input("Enter a number: "))
# print("Input :", input_list)
# print("Sorted:", sorted(input_list))

####################
####  Task 2.2  ####
####################
"""
1. Square all elements of a list.
2. Recursively calculate the sum of all elements in a list.
3.
"""

def square_list(input_list):
    return [x**2 for x in input_list]

# def sum_recursive1(input_list):
#     result = 0
#     for element in input_list:
#         result = result + element
#     return result

# def sum_recursive(input_list, i=0):
#     if i == len(input_list)-1:
#         return input_list[i]
#     else:
#         return input_list[i] + sum_recursive(input_list, i+1)

# def compute_mean(input_list):
#     return sum(input_list) / len(input_list)

# input_list = list(range(1,6))
# print("Input:  ", input_list)
# print("Squared:", square_list(input_list))
# print("Summed: ", sum_recursive(input_list))
# print("Mean:   ", compute_mean(input_list))

####################
####  Task 2.3  ####
####################
"""
Class with
1. Constructor
2. Length function to compute eucledian distance
3. Add function to add two Vec2
4. Variable id and global class variable gid
"""
# import math

# class Vec2:

#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.id = gid
    
#     def length(self):
#         return math.sqrt(self.x**2 + self.y**2)

#     def add(self, rhs):
#         return Vec2(self.x+rhs.x, self.y+rhs.y)

# a = Vec2(0,0)
# b = Vec2(1,1)
# c = Vec2(2,2)

####################
####  Task 2.4  ####
####################
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
    # OpenCV uses BGR but matplot uses RGB format.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt_img = plt.imshow(img)
    if show_window_now:
        plt.show()
    return plt_img

img = cv2.imread("img/hummingbird_from_pixabay.png")
showImage(img)

def imageStats(img):
    print("Image stats:")
    print("Width:   ", img.shape[0])
    print("Height:  ", img.shape[1])
    print("Channels:", img.shape[2])

imageStats(img)

modified_img = img[:,:,::-1] # from:to:step
showImage(modified_img)
cv2.imwrite("img/blue_hummingbird.png", img)

####################
####  Task 2.5  ####
####################
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
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def img_update(hue_offset):
    print("Set hue offset to " + str(hue_offset))
    # Change the hue channel of the HSV image by `hue_offset`.
    # Mind that hue values in OpenCV range from 0-179.
    mask = cv2.inRange(hsv_img[:,:,0], 0, 32) # Found by setting hue_offset

    temp        = cv2.copyTo(hsv_img, mask=mask)
    temp[:,:,0] = temp[:,:,0] + hue_offset

    # Combine changed bird with old background.
    np.where(temp[:,:,0] != 0, temp, hsv_img)

    # Convert the modified HSV image back to RGB and update the image in the plot window using `plt_img.set_data(img_rgb)`.
    rgb_img = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
    plt_img.set_data(rgb_img)

# Create an interactive slider for the hue value offset.
ax_hue = plt.axes([0.1, 0.04, 0.8, 0.06]) # x, y, width, height
slider_hue = Slider(ax=ax_hue, label='Hue', valmin=0, valmax=180, valinit=0, valstep=1)
slider_hue.on_changed(img_update)

# Now actually show the plot window
plt.show()
