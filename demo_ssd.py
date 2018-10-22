"""1. Predict with pre-trained SSD models
==========================================

This article shows how to play with pre-trained SSD models with only a few
lines of code.

First let's import some necessary libraries:
"""
import time
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx


net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
# net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. Here we
# specify that we resize the short edge of the image to 512 px. But you can
# feed an arbitrarily sized image.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.ssd.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print(type(x))
print('Shape of pre-processed image:', x.shape)

######################################################################
# Inference and display
# ---------------------
#
# The forward function will return all detected bounding boxes, and the
# corresponding predicted class IDs and confidence scores. Their shapes are
# `(batch_size, num_bboxes, 1)`, `(batch_size, num_bboxes, 1)`, and
# `(batch_size, num_bboxes, 4)`, respectively.
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:
# num_iter = 1000
# x = mx.ndarray.concat(*[x, x, x, x, x, x, x, x, x, x], dim=0)
# print(x.shape)
# t1 = time.time()
# for i in range(num_iter):
#     class_IDs, scores, bounding_boxs = net(x)
#
# t2 = time.time()
#
#
# print((t2 - t1)/num_iter/10)
# print(class_IDs)
# print(len(class_IDs[0]))

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()
