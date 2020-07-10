import tensorflow as tf
from tensorflow import keras

# Architektúra diskriminátora

class DISCRIM():

	def __init__(self,input_shape,optimizer = None):
		self.model_made = False
		self.model = None
		self.input_shape = input_shape
		self.optimizer  = optimizer
		if optimizer is None:
			self.optimizer = tf.keras.optimizers.Adam(1e-4)


	def calc_apply_gradients(self,tape,loss):
		grads = tape.gradient(loss,self.get_model().trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.get_model().trainable_variables))

	@tf.function
	# Vráti hodnotu chybovej funkcie diskriminátora
	# real_output – výstup diskriminátora, ak na vstupe boli dáta z tréningovej množiny = D(X)
	# fake_output – výstup diskriminátora, ak na vstupe boli dáta generované generátorom = D(G(z))
	def loss(self,real_output, fake_output):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		real_loss = cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss

	# Uloží model
	def saveWeights(self, path):
		self.get_model().save_weights(path)


	def loadWeights(self, path):
		self.get_model().load_weights(path)
	# Vráti výstup siete pri vstupe input
	def output(self, input, is_training=False):
		model = self.get_model()
		return model(input, training=is_training)


	def get_model(self):
		if self.model_made:
			return self.model
		model = tf.keras.Sequential()
		model.add(
			keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', input_shape=self.input_shape))
		model.add(keras.layers.Activation(tf.nn.leaky_relu))
		model.add(keras.layers.BatchNormalization())

		model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'))
		model.add(keras.layers.Activation(tf.nn.leaky_relu))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'))
		model.add(keras.layers.Activation(tf.nn.leaky_relu))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128,activation="relu"))
		model.add(keras.layers.Dropout(0.3))
		model.add(keras.layers.Dense(64,activation="relu"))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(1))
		self.model_made = True
		self.model = model
		return self.model
