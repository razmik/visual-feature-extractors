import numpy as np
from skimage.feature import hog


def to_HoG(data, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)):

    result_data = []
    for img in data:
        # fd.shape = (orientations * img.height * img.width) / (pixels_per_cell * pixels_per_cell)
        feature_vector = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, feature_vector=True, block_norm='L2-Hys')
        result_data.append(feature_vector)

    return np.array(result_data)
