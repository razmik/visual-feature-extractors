import numpy as np
import pandas as pd
import os
import cv2
import csv
import sys
import pySaliencyMap

from os import listdir
from os.path import isdir, isfile, join
from skimage.measure import structural_similarity as compare_ssim
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_image_data(foldername):
    frames_dict = {}
    internal_folders = [f for f in listdir(foldername) if isdir(join(foldername, f))]
    count = 0
    for internal_folder in internal_folders:
        frames_files = [f for f in listdir(join(foldername, internal_folder)) if
                                 isfile(join(foldername, join(internal_folder, f)))]
        image_list = []
        for frame in frames_files:
            file = join(foldername, join(internal_folder, frame))
            image_list.append(np.asarray(cv2.imread(file)))
            frames_dict[internal_folder] = image_list

        count += 1

        # if count > 3:
        #     break

    return frames_dict


def convert_to_model_dataformat(frames_dict, train_test):

    data = []
    data_label = []
    for key, image_array in frames_dict.items():
        data.extend(image_array)
        data_label.extend([str(train_test) + '_' + str(key) + '_' + str(idx+1) for idx, val in enumerate(range(len(image_array)))])

    return np.asarray(data), data_label


def reshape_dataset(input_data, size):

    reshaped_data = []
    for row in [en_img.reshape(size[0], size[1]) for en_img in input_data]:
        reshaped_data.append(row[0])
    return reshaped_data


def conv_saliency_img(input_data):

    sm = pySaliencyMap.pySaliencyMap(input_data.shape[-2], input_data.shape[-1])
    saliency_map = []
    for img in input_data:
        saliency_map.append(sm.SMGetSM(img))

    return np.array(saliency_map)


training_folder_name = "E:/Projects/image-feature-extractor/autoencoder/spatiotemporal_autoencoder/abnormal-spatiotemporal-ae/share/data/videos/ucsd_ped1/training_frames".replace('\\', '/')
testing_folder_name = "E:/Projects/image-feature-extractor/autoencoder/spatiotemporal_autoencoder/abnormal-spatiotemporal-ae/share/data/videos/ucsd_ped1/testing_frames".replace('\\', '/')

print('Read training data')
training_frames_dict = read_image_data(training_folder_name)
testing_frames_dict = read_image_data(testing_folder_name)

print('Convert to training data format')
train_data, train_data_label = convert_to_model_dataformat(training_frames_dict, 'train_')
test_data, test_data_label = convert_to_model_dataformat(testing_frames_dict, 'test_')
test_data_label = np.asarray(test_data_label)

print('Transform to saliency image')
x_train = conv_saliency_img(train_data)
x_test = conv_saliency_img(test_data)

print('Convert to channel first format')
x_train = np.reshape(x_train, (len(x_train), 224, 224, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 224, 224, 1))  # adapt this if using `channels_first` image data format

# Autoencoder
input_img = Input(shape=(224, 224, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='valid', strides=2)(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='valid', strides=2)(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='valid', strides=2)(x)

x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='valid', strides=2)(x)

"""
Input: 224 x 224 x 3
Output: 512
Inspired by: https://arxiv.org/pdf/1409.1556.pdf (Model D)
"""
encoder = Model(input_img, encoded)

x = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
decoded = UpSampling2D((2, 2))(x)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# x_train_test = np.concatenate([x_train, x_test])

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

"""
Save the model
"""
# autoencoder.save('autoencoder.h5')
# encoder.save('encoder.h5')
#
# sys.exit(0)

"""
Load model
"""
# del autoencoder
# del encoder
#
# autoencoder = load_model('autoencoder.h5')
# encoder = load_model('encoder.h5')

"""
Calculate the reconstruction error
"""

predicted_train_imgs = autoencoder.predict(x_train)
predicted_test_imgs = autoencoder.predict(x_test)

# Evaluate error for train data
train_costs = np.zeros(len(x_train))
train_ssim_mean = np.zeros(len(x_train))
train_ssim_max = np.zeros(len(x_train))
for i in range(len(x_train)):

    train_costs[i] = np.linalg.norm(np.squeeze(predicted_train_imgs[i]) - np.squeeze(x_train[i]))

    # ssim_costs = np.zeros(x_train.shape[-1])
    # for j in range(x_train.shape[-1]):
    #     ssim_costs[j] = compare_ssim(predicted_train_imgs[i][:, :, j].astype('float64'), x_train[i][:, :, j])
    # train_ssim_mean[i] = ssim_costs.mean()
    # train_ssim_max[i] = ssim_costs.max()

np.savetxt("hog_train_costs.csv", np.mean(train_costs.reshape(-1, 200), axis=1), delimiter=",")
# np.savetxt("hog_train_costs_ssim_mean.csv", np.mean(train_ssim_mean.reshape(-1, 200), axis=1), delimiter=",")
# np.savetxt("hog_train_costs_ssim_max.csv", np.mean(train_ssim_max.reshape(-1, 200), axis=1), delimiter=",")

# Evaluate error for test data
test_costs = np.zeros(len(x_test))
test_ssim_mean = np.zeros(len(x_test))
test_ssim_max = np.zeros(len(x_test))
for i in range(len(x_test)):

    test_costs[i] = np.linalg.norm(np.squeeze(predicted_test_imgs[i]) - np.squeeze(x_test[i]))

    # ssim_costs = np.zeros(x_test.shape[-1])
    # for j in range(x_test.shape[-1]):
    #     ssim_costs[j] = compare_ssim(predicted_test_imgs[i][:, :, j].astype('float64'), x_test[i][:, :, j])
    # test_ssim_mean[i] = ssim_costs.mean()
    # test_ssim_max[i] = ssim_costs.max()

np.savetxt("hog_test_costs.csv", np.mean(test_costs.reshape(-1, 200), axis=1), delimiter=",")
# np.savetxt("hog_test_costs_ssim_mean.csv", np.mean(test_ssim_mean.reshape(-1, 200), axis=1), delimiter=",")
# np.savetxt("hog_test_costs_ssim_max.csv", np.mean(test_ssim_max.reshape(-1, 200), axis=1), delimiter=",")

"""
Compose the encode directory for GSOM
"""

# encoded_test_imgs = encoder.predict(x_test)
# encoded_train_imgs = encoder.predict(x_train)

# print('processed images to reshaped list')
# encoded_test_imgs = reshape_dataset(encoded_test_imgs, [1, 128])
# encoded_train_imgs = reshape_dataset(encoded_train_imgs, [1, 128])
#
# encoded_test = pd.DataFrame(encoded_test_imgs)
# encoded_test.insert(0, 'index', test_data_label)
#
# encoded_train = pd.DataFrame(encoded_train_imgs)
# encoded_train.insert(0, 'index', train_data_label)
#
# result_dataset = encoded_test.append(encoded_train, ignore_index=True)
#
# result_dataset.to_csv("encoded_dataset_any.csv", header=None, index=None)
# print('saved img and labels')
