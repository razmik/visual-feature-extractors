"""
Source:
https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
"""
# import the necessary packages
from skimage.measure import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = compare_ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()


# load the images -- the original, the original + contrast,
# and the original + photoshop
image1_fpath = '../autoencoder/spatiotemporal_autoencoder/abnormal-spatiotemporal-ae/share/data/videos/ucsd_ped1/training_frames/01/005.jpg'
image2_fpath = '../autoencoder/spatiotemporal_autoencoder/abnormal-spatiotemporal-ae/share/data/videos/ucsd_ped1/training_frames/14/018.jpg'
original_1 = cv2.imread(image1_fpath)
original_2 = cv2.imread(image2_fpath)

# convert the images to grayscale
original_1 = cv2.cvtColor(original_1, cv2.COLOR_BGR2GRAY)
original_2 = cv2.cvtColor(original_2, cv2.COLOR_BGR2GRAY)

# # initialize the figure
# fig = plt.figure("Images")
# images = ("Original 1", original_1), ("Original 2", original_2)
#
# # loop over the images
# for (i, (name, image)) in enumerate(images):
#     # show the image
#     ax = fig.add_subplot(1, 2, i + 1)
#     ax.set_title(name)
#     plt.imshow(image, cmap=plt.cm.gray)
#     plt.axis("off")
#
# # show the figure
# plt.show()

# compare the images
# compare_images(original_1, original_1, "Original 1 vs. Original 1")
compare_images(original_1, original_2, "Original 1 vs. Original 2")
