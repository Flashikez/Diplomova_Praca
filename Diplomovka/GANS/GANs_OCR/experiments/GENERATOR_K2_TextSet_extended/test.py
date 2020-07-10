import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Diplomovka.img_utils as utils

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
image = train_images[0:5]
downsampled = utils.downsample(image,(14,14))
# print(downsampled.shape)
plt.imshow((image[0])[:,:,0],cmap="gray")
plt.show()
plt.imshow((utils.upsample(downsampled,(28,28),tf.image.ResizeMethod.BICUBIC)[0])[:,:,0],cmap="gray")
plt.show()
plt.imshow((downsampled[0])[:,:,0],cmap="gray")
plt.show()