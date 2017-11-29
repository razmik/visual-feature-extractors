"""
Implementation:
https://gurus.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import data, exposure, color, io

# filename = '../../images/bay.jpg'
image = color.rgb2gray(data.astronaut())
# image = data.astronaut()
# image = color.rgb2gray(io.imread(filename))

"""
Normalizing the image prior to description.
This normalization step is entirely optional, but in some cases this step can improve performance of the HOG descriptor. 
The first stage applies an optional global image normalisation equalisation that is designed to reduce the influence of 
illumination effects. In practice we use gamma (power law) compression, either computing the square root or the log of 
each color channel. Image texture strength is typically proportional to the local surface illumination so this 
compression helps to reduce the effects of local shadowing and illumination variations.
"""
# image = np.sqrt(image)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                    visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex='all', sharey='all')

ax1.axis('off')
ax1.imshow(image, cmap='gray')
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap='gray')
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
