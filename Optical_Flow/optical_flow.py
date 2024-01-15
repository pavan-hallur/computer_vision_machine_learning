#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import cv2
from numpy.core.fromnumeric import shape
from scipy import signal

from flow_utils import *
from utils import *



#
# Task 2
#
# Implement Lucas-Kanade or Horn-Schunck Optical Flow.



# TODO: Implement Lucas-Kanade Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# kernel_size: kernel size
# eigen_threshold: threshold for determining if the optical flow is valid when performing Lucas-Kanade
# returns the Optical flow based on the Lucas-Kanade algorithm
def LucasKanadeFlow(frames, Ix, Iy, It, kernel_size, eigen_threshold = 0.01):

    # Create output img.
    img = np.zeros((frames[0].shape[0], frames[0].shape[1], 2))

    # Define window size.
    w_x = kernel_size[0] // 2
    w_y = kernel_size[1] // 2

    for i in range(w_y, frames[0].shape[0] - w_y):
        for j in range(w_x, frames[0].shape[1] - w_x):
            # Get values at current window.
            Ix_window = Ix[i-w_y:i+w_y+1, j-w_x:j+w_x+1].flatten()
            Iy_window = Iy[i-w_y:i+w_y+1, j-w_x:j+w_x+1].flatten()
            It_window = It[i-w_y:i+w_y+1, j-w_x:j+w_x+1].flatten()

            # Solve equation using the pseudo inverse, because A is not a square matrix.
            # Equation: 
            A = np.array([Ix_window, Iy_window]).T
            b = -It_window.T
            pinv = np.linalg.inv((A.T @ A)) @ A.T
            uv = pinv @ b

            # Save u and v.
            img[i,j,0] = uv[0]
            img[i,j,1] = uv[1]

    return img

    
    # # Lecture version.

    # # Compute the harris matrix.
    # harrisMatrix = np.ones((2, 2) + frames[0].shape)
    # # Hint: Each of the following 4 entries contains a full gradient image
    # harrisMatrix[0, 0] = Ix * Ix # Gx^2
    # harrisMatrix[0, 1] = Ix * Iy # Gx*Gy
    # harrisMatrix[1, 0] = Ix * Iy # Gx*Gy
    # harrisMatrix[1, 1] = Iy * Iy # Gy^2
    
    # # Apply smoothing.
    # harrisMatrix[0, 0] = cv2.boxFilter(harrisMatrix[0, 0], -1, kernel_size)
    # harrisMatrix[1, 0] = cv2.boxFilter(harrisMatrix[1, 0], -1, kernel_size)
    # harrisMatrix[0, 1] = cv2.boxFilter(harrisMatrix[0, 1], -1, kernel_size)
    # harrisMatrix[1, 1] = cv2.boxFilter(harrisMatrix[1, 1], -1, kernel_size)

    # # b is the right side of the equation.
    # rightMatrix = np.ones((2, 1) + frames[0].shape)
    # rightMatrix[0,0] = -Ix * It
    # rightMatrix[1,0] = -Iy * It

    # u = np.zeros(frames[0].shape)
    # v = np.zeros(frames[0].shape)

    # w_x = kernel_size[0] // 2
    # w_y = kernel_size[1] // 2

    # for i in range(w_y, frames[0].shape[0] - w_y):
    #     for j in range(w_x, frames[0].shape[1] - w_x):
    #         pass

    # for row in range(frames[0].shape[0]):
    #     for col in range(frames[0].shape[1]):

    #         # Take the 4 entries for the current pixel from the harris matrix.
    #         A = np.array([
    #             [harrisMatrix[0,0,row,col], harrisMatrix[0,1,row,col]],
    #             [harrisMatrix[1,0,row,col], harrisMatrix[1,1,row,col]]
    #         ])

    #         # Take the 2 entries for the current pixel from the right matrix.
    #         b = np.array([
    #             [rightMatrix[0,0,row,col]],
    #             [rightMatrix[1,0,row,col]]
    #         ])

    #         eigenvalues = np.linalg.eigvals(A)
    #         #if not np.any(eigenvalues < eigen_threshold):

    #         # Solve the equation for the current pixel.
    #         uv = np.linalg.inv(A) @ b

    #         # Save u and v
    #         u[row,col] = uv[0,0]
    #         v[row,col] = uv[1,0]

    # # Set u and v in img.
    # img = np.zeros((frames[0].shape[0], frames[0].shape[1], 2))
    # img[:,:,0] = u
    # img[:,:,1] = v
    
    # return img



# TODO: Implement Horn-Schunck Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# max_iterations: maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
# epsilon: the stopping criterion for the difference when performing the Horn-Schuck algorithm
# returns the Optical flow based on the Horn-Schunck algorithm
def HornSchunckFlow(frames, Ix, Iy, It, max_iterations = 1000, epsilon = 0.002):
    return PLACEHOLDER_FLOW(frames)
    #
    # ???
    #



# Load image frames
frames = [  cv2.imread("resources/frame1.png"),
            cv2.imread("resources/frame2.png")]

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("resources/groundTruthOF.flo")

# Grayscales
gray = [(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float64) for frame in frames]

# Get derivatives in X and Y
xdk = np.array([[-1.0, 1.0],[-1.0, 1.0]])
ydk = xdk.T
fx = cv2.filter2D(gray[0], cv2.CV_64F, xdk) + cv2.filter2D(gray[1], cv2.CV_64F, xdk)
fy = cv2.filter2D(gray[0], cv2.CV_64F, ydk) + cv2.filter2D(gray[1], cv2.CV_64F, ydk)

# Get time derivative in time (frame1 -> frame2)
tdk1 = np.ones((2,2))
tdk2 = tdk1 * -1
ft = cv2.filter2D(gray[0], cv2.CV_64F, tdk2) + cv2.filter2D(gray[1], cv2.CV_64F, tdk1)

# Ground truth flow
plt.figure(figsize=(5, 8))
showImages([("Groundtruth flow", flowMapToBGR(flow_gt)),
            ("Groundtruth field", drawArrows(frames[0], flow_gt)) ], 1, False)

# Lucas-Kanade flow
flow_lk = LucasKanadeFlow(gray, fx, fy, ft, [15, 15])
error_lk = calculateAngularError(flow_lk, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("LK flow - angular error: %.3f" % error_lk, flowMapToBGR(flow_lk)),
            ("LK field", drawArrows(frames[0], flow_lk)) ], 1, False)

# Horn-Schunk flow
flow_hs = HornSchunckFlow(gray, fx, fy, ft)
error_hs = calculateAngularError(flow_hs, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("HS flow - angular error %.3f" % error_hs, flowMapToBGR(flow_hs)),
            ("HS field", drawArrows(frames[0], flow_hs)) ], 1, False)

plt.show()
