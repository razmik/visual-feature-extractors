import cv2
import numpy as np

from os import listdir
from os.path import isdir, isfile, join


class DataProcessor:

    @staticmethod
    def read_image_data(foldername, test_dataset=False):
        frames_dict = {}
        internal_folders = [f for f in listdir(foldername) if isdir(join(foldername, f))]
        count = 0
        for internal_folder in internal_folders:
            frames_files = [f for f in listdir(join(foldername, internal_folder)) if
                            isfile(join(foldername, join(internal_folder, f)))]
            image_list = []
            for frame in frames_files:
                file = join(foldername, join(internal_folder, frame))
                image_list.append(np.asarray(cv2.imread(file, 0)))
                frames_dict[internal_folder] = image_list

            count += 1

            if test_dataset:
                if count > 3:
                    break

        return frames_dict

    @staticmethod
    def reshape_dataset(input_data, size):

        reshaped_data = []
        for row in [en_img.reshape(size[0], size[1]) for en_img in input_data]:
            reshaped_data.append(row[0])

        return reshaped_data

    @staticmethod
    def convert_to_model_dataformat(frames_dict, train_test):

        data = []
        data_label = []
        for key, image_array in frames_dict.items():
            data.extend(image_array)
            data_label.extend([str(train_test) + '_' + str(key) + '_' + str(idx + 1) for idx, val in
                               enumerate(range(len(image_array)))])

        return np.asarray(data), data_label

