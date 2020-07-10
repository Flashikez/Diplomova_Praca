import tensorflow as tf
import Diplomovka.img_utils as img
from Diplomovka.Classifiers.K_F_MNIST.K_F_MNIST import K_F_MNIST
from Diplomovka.GANs.architectures.Generators.GENERATOR_K4 import GENERATOR_K4
import Diplomovka.GANs.Utils.metrics_calculator as calc



#Skrip vypočíta metriky pre natrénovaný genrátor v tomto priečinku
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

train_images = img.normalize(train_images)
train_original = tf.identity(train_images)
train_fake = img.downsample(train_images,(14,14))

test_images = img.normalize(test_images)
test_original = tf.identity(test_images)
test_fake = img.downsample(test_images,(7,7))
batch_size = 32


def test_gen():
	for image,label in zip(test_images,test_labels):
		label = tf.one_hot(label,10)
		downsampled = img.downsample(image,(7,7))
		yield image,downsampled,label


dataset_test_fake = tf.data.Dataset.from_tensor_slices(test_fake).batch(16)
dataset_test_real = tf.data.Dataset.from_tensor_slices(test_original).batch(16)

test_datasets_methods = tf.data.Dataset.from_generator(test_gen,(tf.float32,tf.float32,tf.int64)).batch(16)


generator = GENERATOR_K4([7,7,1])
generator.loadWeights('pretrained_models/generator/generator')


classifier = K_F_MNIST([28,28,1])
classifier.loadWeights('../../../../../Diplomovka/Classifiers/K_F_MNIST/pretrained_model/MNIST/classifier')
# accuracy = calc.(generator,classifier,classifier_dataset)
# print(accuracy)

calc.test_generator_v2(generator,classifier,test_datasets_methods)
# ---------------------------------------------------------------------------------
# Testovanie ostatných zväčšovacích metód, vypočíta hodnoty metrík na datasete
# calc.test_upscale_method(tf.image.ResizeMethod.NEAREST_NEIGHBOR,classifier,test_datasets_methods,'NEAREST')
# calc.test_upscale_method(tf.image.ResizeMethod.BILINEAR,classifier,test_datasets_methods,'BILINEAR')
# calc.test_upscale_method(tf.image.ResizeMethod.BICUBIC,classifier,test_datasets_methods,'BICUBIC')
# calc.test_upscale_method(tf.image.ResizeMethod.LANCZOS3 ,classifier,test_datasets_methods,'LANCZOS3')
# calc.test_upscale_method(tf.image.ResizeMethod.LANCZOS5 ,classifier,test_datasets_methods,'LANCZOS5')
