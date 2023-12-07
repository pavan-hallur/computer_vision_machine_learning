#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 3
#
# Load and use a pretrained model to estimate the depth ot a single image.


import cv2
import numpy as np


# TODO: Compatibility
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import sys
sys.path.insert(0, "FCRN-DepthPrediction/tensorflow") # TODO: Adapt to your download location
import models



# Download and unzip a checkpoint containing pre-trained weights.
checkpoint_path = download_checkpoints("NYU_FCRN.ckpt", "NYU_FCRN-checkpoint.zip")

# Load the test image.
img = cv2.imread("livingroom.jpeg")



# Model input size.
height = 228
width = 304
channels = 3
batch_size = 1

# Create a placeholder for the input image.
input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

# Construct the network.
net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

with tf.Session() as sess:

    # Load weights from checkpoint file.
    print('Loading the model weights')
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # TODO: Prepare image for TF.
    #
    # ???
    #

    # TODO: Evalute the network for the given image.
    #  Convert the resulting tensor into a (single channel) depth map as in Task 2, i.e. black for close pixels and white for pixels far away.
    depth = REPLACE_THIS(np.zeros((height, width), np.float32))
    #
    # ???
    #


    # Visualization
    plt.figure(figsize=(14, 4))
    showImages([("img", img), ("depth", depth)], 4, show_window_now=False, padding=[.01, .01, .01, .1])

    plot3D(depth, (2, 4), (0, 2), (2, 2))
    plt.show()
