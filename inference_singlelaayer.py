
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.particle import particle

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# %%

config = particle.ParticleConfig()
PARTICLE_DIR = os.path.join(ROOT_DIR, "..\datasets\particle")


# %%

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_RESIZE_MODE = "square"
    # IMAGE_MIN_DIM = 800
    # IMAGE_MAX_DIM = 1024


config = InferenceConfig()
config.display()

# %% md

## Notebook Preferences

# %%

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
# DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
DEVICE = "/gpu:0"

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# %%

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# %% md

## Load Validation Dataset

# %%

# Load validation dataset
dataset = particle.ParticleDataset()
dataset.load_particle(PARTICLE_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# %% md

## Load Model

# %%

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# %%

# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
weights_path = "C:\Project\logs\particle20190725T2129\mask_rcnn_particle_0133.h5"

# weights_path = "C:\Project\logs\particle20190815T1512\mask_rcnn_particle_0111.h5"

# Or, load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# %%


# %% md

# %%

import skimage.draw
from PIL import Image
import matplotlib.pyplot as plt
from numpy import zeros, newaxis


def pyramidDetect(image, savefile):
    allMASK = np.zeros([image.shape[0], image.shape[1]]).astype(np.uint8)
    image0 = image
    for i in np.arange(0, 40.0, 1):
        # for i in np.arange(0, 1, 1):
        fac = 0.1 * (i + 1)
        # fac = 1
        image = image0
        image = np.array(
            Image.fromarray(image).resize([int(np.round(image.shape[1] * fac)), int(np.round(image.shape[0] * fac))]))

        results = model.detect([image], verbose=0)

        # Display results
        r = results[0]
        # ax = get_ax(1)

        mask = r['masks']
        MASK = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
        for c in range(len(mask[0, 0])):
            if (mask[:, :, c].sum() / len(mask[:, :, c]) < 10):
                MASK = MASK + mask[:, :, c] * (c + 1) * 1

        allMASK = allMASK + np.array(Image.fromarray(MASK).resize([image0.shape[1], image0.shape[0]]))

        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                   dataset.class_names, r['scores'], ax=ax,
        #                    title="Predictions")

        # plt.imshow(allMASK)
        # plt.show()

    masked_image = Image.fromarray(allMASK)

    # plt.subplot(121)
    # plt.imshow(image0)
    # plt.subplot(122)
    # plt.imshow(allMASK)
    # plt.show()

    masked_image.save(savefile)

#
path = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\LithB_Proj1_NMC811_sample1A_100nm_fasttomo_2.to-byte_512\\"
savepath = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\mask_LithB_Proj1_NMC811_sample1A_100nm_fasttomo_2.to-byte_512\\"
valid_images = [".jpg"]
for f in os.listdir(path):
    print(f)
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue

    image = skimage.io.imread(os.path.join(path, f))

    image = skimage.color.gray2rgb(image)

    savefile = os.path.join(savepath, '%s.tif' % (name))
    print(savefile)
    pyramidDetect(image, savefile)

# %%
# image = skimage.io.imread("G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\LithB_Proj1_NMC811_sample1A_100nm_fasttomo_2.to-byte_512\\slice_0138.jpg")
# savefile = "C:\\Users\\Jizhou Li\\Desktop\\test.tif"
# image = skimage.color.gray2rgb(image)
# pyramidDetect(image, savefile)


path = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\single_1B_T1\\"
savepath = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\mask_single_1B_T1\\"
valid_images = [".jpg"]
for f in os.listdir(path):
    print(f)
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue

    image = skimage.io.imread(os.path.join(path, f))

    image = skimage.color.gray2rgb(image)

    savefile = os.path.join(savepath, '%s.tif' % (name))
    print(savefile)
    pyramidDetect(image, savefile)

path = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\single_1B_T2\\"
savepath = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\mask_single_1B_T2\\"
valid_images = [".jpg"]
for f in os.listdir(path):
    print(f)
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue

    image = skimage.io.imread(os.path.join(path, f))

    image = skimage.color.gray2rgb(image)

    savefile = os.path.join(savepath, '%s.tif' % (name))
    print(savefile)
    pyramidDetect(image, savefile)

path = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\single_2A\\"
savepath = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\mask_single_2A\\"
valid_images = [".jpg"]
for f in os.listdir(path):
    print(f)
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue

    image = skimage.io.imread(os.path.join(path, f))

    image = skimage.color.gray2rgb(image)

    savefile = os.path.join(savepath, '%s.tif' % (name))
    print(savefile)
    pyramidDetect(image, savefile)

path = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\single_2B\\"
savepath = "G:\\SLAC\\201907_slac_LIBNet\\ESRF machine learning\\mask_single_2B\\"
valid_images = [".jpg"]
for f in os.listdir(path):
    print(f)
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue

    image = skimage.io.imread(os.path.join(path, f))

    image = skimage.color.gray2rgb(image)

    savefile = os.path.join(savepath, '%s.tif' % (name))
    print(savefile)
    pyramidDetect(image, savefile)