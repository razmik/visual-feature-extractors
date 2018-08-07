"""
Tutorial:
https://blog.keras.io/building-autoencoders-in-keras.html
https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import hog

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.datasets import mnist
# from skimage.feature import hog

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Stack several layers of hidden layers.
"""

#  We're using MNIST digits, and
# we're discarding the labels (since we're only interested in encoding/decoding the input images)
print('Loading MINIST Data')
(x_train, _), (x_test, y_test) = mnist.load_data()

print('Computing HoG')
x_train = hog.to_HoG(x_train)
x_test = hog.to_HoG(x_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 7, 7, 8))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 7, 7, 8))  # adapt this if using `channels_first` image data format
print("Train data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# this is our input placeholder
input_img = Input(shape=(7, 7, 8))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_2')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

encoder = Model(input_img, encoded)

x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_4')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_6')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(8, (2, 2), activation='sigmoid', name='conv_7')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# Save encoded images with its labels
encoded_imgs_reshaped_processed = []
for row in [en_img.reshape(1, 4*8) for en_img in encoded_imgs]:
    encoded_imgs_reshaped_processed.append(row[0])

print('processed images to reshaped list')

with open("encoded_imgs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(encoded_imgs_reshaped_processed)

print('saved img codes')

np.savetxt("encoded_imgs_labels.csv", y_test, delimiter=",")

print('saved img labels')

# Display results

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
plt.title("Plot 2")
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(49, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(4, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(49, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()