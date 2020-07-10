import tensorflow as tf

# Vypočíta presnosť klasifikátora na zadanom datasete
def accuracy_on_dataset(classifier, test_dataset):
	correct_predictions = 0
	wrong_predictions = 0
	for test_images, test_labels in test_dataset:
		# print("batch")
		predicted = classifier.output(test_images, training=False)
		real_classes = tf.keras.backend.argmax(test_labels, axis=1)
		predicted_classes = tf.keras.backend.argmax(predicted, axis=1)
		correct_predictions += tf.keras.backend.sum(
			tf.keras.backend.cast(tf.keras.backend.equal(real_classes, predicted_classes), tf.float32))
		wrong_predictions += tf.keras.backend.sum(
			tf.cast(tf.keras.backend.not_equal(real_classes, predicted_classes), tf.float32))
	return correct_predictions / (correct_predictions + wrong_predictions)

# Vypočíta presnost klsifikátora na obrázkoch zväčšených generátorom generator
def accuracy_on_dataset_with_generator(generator,classifier, test_dataset):
	correct_predictions = 0
	wrong_predictions = 0
	for test_images, test_labels in test_dataset:
		# print("batch")
		test_images = generator.output(test_images,is_training=False)
		predicted = classifier.output(test_images, training=False)
		real_classes = tf.keras.backend.argmax(test_labels, axis=1)
		predicted_classes = tf.keras.backend.argmax(predicted, axis=1)
		correct_predictions += tf.keras.backend.sum(
			tf.keras.backend.cast(tf.keras.backend.equal(real_classes, predicted_classes), tf.float32))
		wrong_predictions += tf.keras.backend.sum(
			tf.cast(tf.keras.backend.not_equal(real_classes, predicted_classes), tf.float32))
	return correct_predictions / (correct_predictions + wrong_predictions)


# Vypočíta hodnotu chybovej funkcie klasifikátora na datasete
def loss_on_dataset(classifier,test_dataset):
	total_loss = 0
	for test_images,real_labels in test_dataset:
		predicted = classifier.output(test_images, training=False)
		total_loss += classifier.loss(real_labels,predicted)
	return total_loss

# Appendne riadok do súboru
def append_to_file(path,text):
	import os
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path,'a') as f:
		f.write(text)
