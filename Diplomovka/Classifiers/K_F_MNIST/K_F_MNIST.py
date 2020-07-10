import tensorflow as tf
from tensorflow import keras

# Architektúra klasifikátora
class K_F_MNIST():
	def __init__(self,input_size,optimizer=None):
		self.input_size = input_size
		self.model_made = False
		self.model = None
		self.layer_output_model = None
		self.layer_output_model_made = False
		if optimizer is None:
			# Vytvorí ADAM optimalizátor na úpravu váh modelu na základe gradientov
			self.optimizer = tf.keras.optimizers.Adam(0.001)

	# Použije gradienty nachdádzajúce sa v GradientTape tape na úpravu váh modelu podľa konkrétneho optimalizačného algoritmu, v tomto prípade ADAM
	# tape - GradientTape, ktorý obsahuje pohľad na gradienty tohto modelu
	# loss - hodnota chybovej funkcie, ktorá sa použije na výpočet hodnôt konkrétnych gradientov
	def calc_apply_gradients(self, tape, loss):
		grads = tape.gradient(loss, self.get_model().trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.get_model().trainable_variables))

	@tf.function
	# Vráti hodnotu chybovej funkcie, v tomto prípade Krížová entropia
	# real_labels - one hot vektor, skutočnej kategórie vstupu, napríklad pre číslicu 5 [0,0,0,0,0,1,0,0,0,0]
	# my labels - pravedpodobnostný vektor, ktorý je výstupom tejto siete pre konkrétny vstup, teda pred tým bola volaná metóda output()
	def loss(self, real_labels, my_labels):
		cross_entropy = tf.keras.losses.CategoricalCrossentropy()
		loss = cross_entropy(real_labels, my_labels)
		return loss

	@tf.function
	def accuracy(self, real_labels, my_labels):
		real_classes = tf.keras.backend.argmax(real_labels, axis=1)
		predicted_classes = tf.keras.backend.argmax(my_labels, axis=1)
		accurate = tf.keras.backend.equal(real_classes, predicted_classes)
		accuracy = tf.keras.backend.mean(accurate)
		return accuracy

	# Vráti výstup neurónovej siete
	# input - vstup do neurónovej siete(modelu)
	# training=True - ak výstup počítame vo fáze tréningu siete
	def output(self, input, training=True):
		model = self.get_model()
		return model(input, training=training)

	# Vytvorí iný pohľad na celkový model, ktorého výstup bude výstup z layer_num vrstvy
	# layer_num – index vrstvy modelu
	def make_hidden_layer_model(self,layer_num):
		if self.layer_output_model_made:
			return self.layer_output_model

		model = self.get_model()
		print(model.layers)
		output_layer = model.layers[layer_num]
		# print('Name of output layer:' + output_layer.name)
		output_model = tf.keras.Model(inputs=model.input,outputs =output_layer.output)
		self.layer_output_model = output_model

	# Vráti výstup (aktivácie) pohľadu na model vytvoreného v metóde make_hidden_layer_model
	# input – vstupný vektor do neurónovej siete.
	def hidden_layer_output(self,input):
		return self.layer_output_model(input)

	def saveWeights(self,path):
		self.get_model().save_weights(path)

	def loadWeights(self,path):
		self.get_model().load_weights(path)




	def get_model(self):
		if self.model_made:
			return self.model
		model = tf.keras.Sequential()

		model.add(
			keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=self.input_size))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.MaxPooling2D(pool_size=2))
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.MaxPooling2D(pool_size=2))
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.MaxPooling2D(pool_size=2))
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(64, activation='relu'))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(32, activation='relu'))
		model.add(keras.layers.Dropout(0.1))
		model.add(keras.layers.Dense(10, activation='softmax'))
		self.model_made = True
		self.model = model
		return self.model