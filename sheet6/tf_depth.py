#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#
# Task 3
#
# Load and use a pretrained model to estimate the depth ot a single image.


from PIL.Image import ANTIALIAS
import cv2
import numpy as np

from tf_utils import *
from depth_utils import *
from utils import *

# TODO: Compatibility
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import sys
sys.path.insert(0, "FCRN-DepthPrediction/tensorflow") # TODO: Adapt to your download location
import models



# Download and unzip a checkpoint containing pre-trained weights.
checkpoint_path = download_checkpoints("NYU_FCRN.ckpt", "NYU_FCRN-checkpoint.zip")

# Load the test image.
img = cv2.imread("img/livingroom.jpeg")


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
    print("img shape:       ", img.shape)
    # Resize for model size.
    input_img = cv2.resize(img, dsize=(width,height))
    # Image needs to be float32.
    input_img = input_img.astype("float32")
    # Expand for batch dimension
    input_img = np.expand_dims(input_img, axis = 0)
    print("input_img shape: ", input_img.shape)


    # TODO: Evalute the network for the given image.
    prediction = sess.run(net.get_output(), feed_dict={input_node: input_img})
    print("prediction shape:", prediction.shape)


    # TODO: Convert the resulting tensor into a (single channel) depth map as in Task 2
    # Remove batch dimension.
    depth = prediction[0,:,:,0]
    print("depth shape:     ", depth.shape)
    # Prepare depth for visualization.
    h_small = height // 4
    w_small = width  // 4
    depth = cv2.medianBlur(depth, 5)
    depth = cv2.resize(depth, dsize=(w_small, h_small), interpolation=cv2.INTER_AREA)


    # Visualization
    plt.figure(figsize=(14, 4))
    showImages([("img", img), ("depth", depth)], 4, show_window_now=False, padding=[.01, .01, .01, .1])

    plot3D(depth, (2, 4), (0, 2), (2, 2))
    plt.show()
