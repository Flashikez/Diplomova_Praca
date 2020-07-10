import tensorflow as tf
from tensorflow.keras import layers

# Architektura generatora, ktory v experimetoch realizuje štvornásobné zvačš1enie rozlisenia
class GENERATOR_K4():

	def __init__(self,input_shape,optimizer=None):
		self.model_made = False
		self.model = None
		self.input_shape = input_shape
		self.optimizer = optimizer
		if optimizer is None:
			self.optimizer = tf.keras.optimizers.Adam(1e-4)



	def calc_apply_gradients(self,tape,loss):
		grads = tape.gradient(loss,self.get_model().trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.get_model().trainable_variables))

	def output(self,input,is_training=False):
		model = self.get_model()
		return model(input,training=is_training)


	def saveWeights(self,path):
		self.get_model().save_weights(path)

	def loadWeights(self, path):
		self.get_model().load_weights(path)
		self.model_made = True

	@tf.function
	# Vráti hodnotu chybovej funkcie generátora
	# fake_output - výstup diskriminátora ak na jeho vstupe bol výstup z generátora, teda hodnota D(G(z))
	def loss(self, fake_output):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		return cross_entropy(tf.ones_like(fake_output),fake_output)



	def get_model(self):
		if self.model_made:
			return self.model

		model = tf.keras.Sequential()
		model.add(layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='valid',
										 input_shape=self.input_shape))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())
		# model.add(layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='valid'))
		#
		# assert model.output_shape == (None, 28, 28, 256)
		# assert model.output_shape == (None, 14, 14, 256)

		model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())
		# assert model.output_shape == (None, 14, 14, 128)

		#
		model.add(layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same'))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())
		print(model.output_shape)
		# assert model.output_shape == (None, 28, 28, 256)

		model.add(layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='same'))
		# assert model.output_shape == (None, 28, 28, 64)
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same'))
		# assert model.output_shape == (None, 28, 28, 32)
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(1, kernel_size=(2, 2), strides=(1, 1), padding='same',
								activation='tanh'))
		# assert model.output_shape == (None, 28, 28, 1)


		self.model_made = True
		self.model = model
		return self.model
