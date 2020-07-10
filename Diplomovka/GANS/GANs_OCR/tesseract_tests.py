import string
import numpy as np
import Diplomovka.Classifiers.Classifier_utils as class_utils
import Diplomovka.GANs.GANs_OCR.tesseract as tess
import tensorflow as tf


# Otestuje dataset na OCR, výsledky zapíše do súbora
def accuracy_on_dataset(dataset,dataset_size,batch_size,log_file_name,log_file_path='tesseract_logs/'):
	print(f'Starting test {log_file_name}')
	correct = 0
	batch_n = 1
	for batch in dataset:

		print(f"Doing batch: {batch_n} of {dataset_size // batch_size} ")
		images, labels = batch[:]

		for image, label in zip(images, labels):

			gray = image[:, :, 0] * 127.5 + 127.5
			gray = tf.round(gray)
			gray = tf.clip_by_value(gray, 0, 255)
			gray = np.uint8(gray)


			label = label.numpy()[:-4]
			label = label.decode()

			threshholded = tess.threshhold(gray)

			ocred_text = tess.ocr_text(threshholded)

			ocred_text = tess.format_as_ecv(ocred_text)

			if ocred_text == label:
				correct += 1
		print(f"Correct after batch {batch_n} : {correct} out of {batch_size * batch_n}\n---------------------------------\n")
		batch_n = batch_n + 1

	#
	print('Test ended. Correct predictions: ', correct, f'Accuracy is: {correct/dataset_size}%')
	print(f'Saving to file {log_file_path}{log_file_name}.txt ')
	class_utils.append_to_file(f"{log_file_path}{log_file_name}.txt", f"correct_predictions:accuracy\n")
	class_utils.append_to_file(f"{log_file_path}{log_file_name}.txt", f"{correct}:{correct/dataset_size}\n")


# Otestuje dataset na OCR a zaznamenáva aj počrty správne/nesprávne rozpzonaných jednotlivých znakov, výsledky zapíše do súbora
def accuracy_on_dataset_log_alphabet(dataset, dataset_size, batch_size, log_file_name, log_file_path='tesseract_logs/'):
	print(f'Starting test {log_file_name}')
	correct = 0
	batch_n = 1
	alphabet_dict = dict((letter,[0,0,0]) for letter in string.ascii_uppercase)
	other_chars_dict = {' ':[0,0,0],'0':[0,0,0],'1':[0,0,0],'2':[0,0,0],'3':[0,0,0],'4':[0,0,0],'5':[0,0,0],'6':[0,0,0],'7':[0,0,0],'8':[0,0,0],'9':[0,0,0]}
	tracking_dict = dict(list(alphabet_dict.items())+list(other_chars_dict.items()))

	print(tracking_dict)
	for batch in dataset:
		# if batch_n == 2:
		# 	break
		print(f"Doing batch: {batch_n} of {dataset_size // batch_size} ")
		images, labels = batch[:]

		for image, label in zip(images, labels):
			# print(original)
			# image = tess.numpy_data_bw(image)
			gray = image[:, :, 0] * 127.5 + 127.5
			gray = tf.round(gray)
			gray = tf.clip_by_value(gray, 0, 255)
			gray = np.uint8(gray)

			label = label.numpy()[:-4]
			label = label.decode()

			threshholded = tess.threshhold(gray)

			ocred_text = tess.ocr_text(threshholded)
			ocred_text = tess.format_as_ecv(ocred_text)
			# print('OCRED TEXT: ',ocred_text, 'ORIGINAL: ',label)
			for i,char in enumerate(label):

				(tracking_dict[char])[0] += 1

				if len(label) == len(ocred_text):
					if ocred_text[i] == label[i]:
						(tracking_dict[char])[1] += 1

					else:
						(tracking_dict[char])[2] += 1

			if ocred_text == label:
				correct += 1
		print(
			f"Correct after batch {batch_n} : {correct} out of {batch_size * batch_n}\n---------------------------------\n")
		batch_n = batch_n + 1

	#
	print('Test ended. Correct predictions: ', correct, f'Accuracy is: {correct / dataset_size}%')
	print(f'Saving to file {log_file_path}{log_file_name}.txt ')
	class_utils.append_to_file(f"{log_file_path}/ACCURACY/{log_file_name}.txt", f"correct_predictions:accuracy\n")
	class_utils.append_to_file(f"{log_file_path}/ACCURACY/{log_file_name}.txt", f"{correct}:{correct / dataset_size}\n")
	class_utils.append_to_file(f"{log_file_path}/CHARS_STATS/{log_file_name}.txt",f"character:TOTAL:CORRECT:INCORRECT\n")
	for k,v in tracking_dict.items():
		class_utils.append_to_file(f"{log_file_path}/CHARS_STATS/{log_file_name}.txt",f"{k}:{v[0]}:{v[1]}:{v[2]}\n")