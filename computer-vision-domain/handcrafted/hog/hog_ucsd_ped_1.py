import numpy as np
import os
import sys
import hog
sys.path.append('../')

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from util import evaluate_regularity as evaluator
from util import data_processor as data_processor


training_folder_name = "E:/Projects/image-feature-extractor/autoencoder/spatiotemporal_autoencoder/abnormal-spatiotemporal-ae/share/data/videos/ucsd_ped1/training_frames".replace('\\', '/')
testing_folder_name = "E:/Projects/image-feature-extractor/autoencoder/spatiotemporal_autoencoder/abnormal-spatiotemporal-ae/share/data/videos/ucsd_ped1/testing_frames".replace('\\', '/')
save_path = 'output/'

if __name__ == '__main__':

    print('Read training data')
    training_frames_dict = data_processor.DataProcessor.read_image_data(training_folder_name)
    testing_frames_dict = data_processor.DataProcessor.read_image_data(testing_folder_name)

    print('Convert to training data format')
    train_data, train_data_label = data_processor.DataProcessor.convert_to_model_dataformat(training_frames_dict, 'train')
    test_data, test_data_label = data_processor.DataProcessor.convert_to_model_dataformat(testing_frames_dict, 'test')
    test_data_label = np.asarray(test_data_label)

    print('Convert to HOG')
    x_train = hog.to_HoG(train_data, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    x_test = hog.to_HoG(test_data, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))

    print('Convert to channel first format')
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 8))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 8))  # adapt this if using `channels_first` image data format

    # Autoencoder
    input_img = Input(shape=(28, 28, 8))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    encoder = Model(input_img, encoded)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # Train the autoencoder with both train and test data
    # x_train_test = np.concatenate([x_train, x_test])

    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    predicted_train_imgs = autoencoder.predict(x_train)
    predicted_test_imgs = autoencoder.predict(x_test)

    # Calculate the reconstruction error
    evaluator.Evaluation.evaluate_error_costs(predicted_train_imgs, x_train, 200, 0, save_path, 'train')
    evaluator.Evaluation.evaluate_error_costs(predicted_test_imgs, x_test, 200, 0, save_path, 'test')

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
