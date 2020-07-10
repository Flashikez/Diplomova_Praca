import tensorflow as tf
import Diplomovka.img_utils as img
from Diplomovka.Classifiers.K_F_MNIST.K_F_MNIST import K_F_MNIST
from Diplomovka.GANs.architectures.Generators.GENERATOR_K4 import GENERATOR_K4
from Diplomovka.GANs.architectures.Discriminators.DISCRIM import DISCRIM
from Diplomovka.GANs.trainers.GAN_Best_Classic_loss_Trainer import GAN_Best_classic_loss_trainer as Trainer


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

train_images = img.normalize(train_images)
train_original = tf.identity(train_images)
train_fake = img.downsample(train_images,(7,7))

test_images = img.normalize(test_images)
test_original = tf.identity(test_images)
test_fake = img.downsample(test_images,(7,7))
batch_size = 32


def test_gen():
	for img,label in zip(test_fake,test_labels):
		label = tf.one_hot(label,10)

		yield img,label

classifier_dataset = tf.data.Dataset.from_generator(test_gen,(tf.float32,tf.int64)).batch(batch_size)
dataset_test_fake = tf.data.Dataset.from_tensor_slices(test_fake).batch(16)
dataset_test_real = tf.data.Dataset.from_tensor_slices(test_original).batch(16)


dataset_fake = tf.data.Dataset.from_tensor_slices(train_fake).batch(batch_size)
dataset_real = tf.data.Dataset.from_tensor_slices(train_original).batch(batch_size)

discriminator = DISCRIM([28,28,1])
generator = GENERATOR_K4([7,7,1])

classifier = K_F_MNIST([28,28,1])
# v2/DCGANs/Diplomovka/Classifiers/K_MNIST_1/pretrained_model/classifier.index
classifier.loadWeights('../../../../../Diplomovka/Classifiers/K_F_MNIST/pretrained_model/FMNIST/classifier')
trainer = Trainer(dataset_real,dataset_fake,dataset_test_fake,dataset_test_real,classifier_dataset,generator,discriminator,classifier,batch_size)
trainer.start('pretrained_models/generator/','pretrained_models/discriminator/',plot_title='Tréning GENERATOR_K4 - klasická chybová funkcia')