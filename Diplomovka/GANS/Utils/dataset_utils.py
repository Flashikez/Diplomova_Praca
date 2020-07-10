import os
import tensorflow as tf
import random

# Vytvorí dataset z priečinka obrázkov, kde anotácia(label) obrázka je názov súboru
def make_dataset(path,batch_size,normalize=True,shuffle=True,seed=None):

	data_dir = os.path.abspath(path)
	print(data_dir)

	if shuffle:
		if seed is None:
			seed = random.randint(0, 4294967295)

		list_ds = tf.data.Dataset.list_files(str(data_dir + '/*'),shuffle=True,seed=seed)
	else:
		list_ds = tf.data.Dataset.list_files(str(data_dir + '/*'), shuffle=False)

	@tf.function
	def get_label(file_path):

		parts = tf.strings.split(file_path, '\\')

		label = parts[-1]

		return label

	@tf.function
	def decode_img(img):
		# convert the compressed string to a 3D uint8 tensor
		image = tf.io.decode_jpeg(img, channels=1)
		image = tf.keras.backend.cast(image,tf.float32)
		if normalize:
			image = (image - 127.5) / 127.5

		return image

	@tf.function
	def process_path(file_path):

		label = get_label(file_path)

		img = tf.io.read_file(file_path)
		img = decode_img(img)
		return img, label

	labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	labeled_ds = labeled_ds.batch(batch_size)
	return labeled_ds
