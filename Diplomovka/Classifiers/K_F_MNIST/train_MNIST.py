import tensorflow as tf
from Diplomovka.Classifiers.K_F_MNIST.K_F_MNIST import K_F_MNIST
from Diplomovka.Classifiers.Classifier_Trainer import Classifier_Trainer as Trainer
import Diplomovka.Classifiers.Classifier_utils as utils


# Skript natrénuje architektúru K_F_MNIST na datasete MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def normalize(imgs):
	return (imgs - 127.5) / 127.5
# Upravi tenzor na tvar [POCET,28,28,1]
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# Normalizuje hodnoty pixelov do -1;1
train_images = (train_images - 127.5) / 127.5

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5

def train_gen():
	for img,label in zip(train_images,train_labels):
		label = tf.one_hot(label, 10)
		yield img,label

def test_gen():
	for img,label in zip(test_images,test_labels):
		label = tf.one_hot(label,10)

		yield img,label

batch_size = 32

train_dataset = tf.data.Dataset.from_generator(train_gen,(tf.float32,tf.int64)).batch(batch_size)
test_dataset = tf.data.Dataset.from_generator(test_gen,(tf.float32,tf.int64)).batch(batch_size)

classifier = K_F_MNIST([28,28,1])
trainer = Trainer(train_dataset,test_dataset,classifier,batch_size)
trainer.start("pretrained_model/MNIST/",save_name="classifier",epochs=8,log_path='training_log/MNIST/')

# classifier.loadWeights("pretrained_model/MNIST/classifier")
# acc = utils.accuracy_on_dataset(classifier,test_dataset)
# print(acc)
