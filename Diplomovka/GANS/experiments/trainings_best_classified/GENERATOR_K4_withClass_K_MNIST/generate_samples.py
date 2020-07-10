import tensorflow as tf
import Diplomovka.img_utils as img
from Diplomovka.Classifiers.K_F_MNIST.K_F_MNIST import K_F_MNIST
from Diplomovka.GANs.architectures.Generators.GENERATOR_K4 import GENERATOR_K4
import Diplomovka.GANs.Utils.metrics_calculator as calc
from Diplomovka.GANs.trainers.GAN_Best_Classified_Trainer import GAN_classic_loss_trainer as Trainer
from Diplomovka.GANs.Utils.GAN_Progress_Maker import Progress_Maker


# Skript vygeneruje príklady zväčšenia generátorom alebo inou zväčšovacou metódou
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

batch = 8
test_datasets_methods = tf.data.Dataset.from_generator(test_gen,(tf.float32,tf.float32,tf.int64)).batch(batch)


generator = GENERATOR_K4([7,7,1])
generator.loadWeights('../../trainings_best_classified_percp_loss/GENERATOR_K4_withClass_K_MNIST_Percp/pretrained_models/generator/generator')
# generator.loadWeights('pretrained_models/generator/generator')
num_of_batches_to_plot = 4
progress_maker = Progress_Maker('Príklad zväčšených obrázkov MNIST LANCZOS5 interpoláciou ',batch,num_of_batches_to_plot*3)


count = 0
skip_first= True
for batch in test_datasets_methods:
	origs,downsampled,_ = batch[:]
	if skip_first:
		skip_first = False
		continue
	# upsampled = generator.output(downsampled,is_training=False)
	upsampled = img.upsample(downsampled,(28,28),tf.image.ResizeMethod.LANCZOS5)
	progress_maker.add_batch(origs,'Originál',)
	progress_maker.add_batch(downsampled, 'Zmenšené')
	progress_maker.add_batch(upsampled, 'Zväčšené')
	count +=1
	if count == num_of_batches_to_plot:
		break
progress_maker.save_progress('examples/','examples_LANCZOS5')


