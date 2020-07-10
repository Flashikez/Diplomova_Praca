import Diplomovka.GANs.Utils.metrics as metrics
import Diplomovka.Classifiers.Classifier_utils as class_utils
import tensorflow as tf
import Diplomovka.img_utils as img

# Vypočíta hodnoty porovnávacích metrík pre zvacseny dataset s originalym nezmensenym datasetom, ktoré následne zapíše do súboru
def test_generator(generator,test_dataset_fake,test_dataset_real,log_path='log/methods_test/tests/'):
	print('Starting Generator Test')
	total_mse = 0
	total_ssim = 0
	total_psnr = 0
	total_euclid = 0
	total_manhattan = 0
	batches_done = 0

	for test_batch, orig_batch in zip(test_dataset_fake, test_dataset_real):

		batch_size = test_batch.shape[0]
		generated = generator.output(test_batch,is_training=False)

		total_manhattan += metrics.manhattan_distance(orig_batch, generated)
		total_mse += metrics.mean_square_error(orig_batch, generated)
		total_ssim += metrics.SSIM(orig_batch, generated)
		total_psnr += metrics.PSNR(orig_batch, generated)
		total_euclid += metrics.euclid_distance(orig_batch, generated)
		batches_done += 1

	total_samples = batches_done*batch_size
	mse = total_mse / total_samples
	euclid = total_euclid / total_samples
	psnr = total_psnr / total_samples
	ssim = total_ssim/ total_samples
	print('Generator tested')

	class_utils.append_to_file(log_path + "generator.txt", f'mse:{mse}\n')
	class_utils.append_to_file(log_path + "generator.txt", f'euclid:{euclid}\n')
	class_utils.append_to_file(log_path + "generator.txt", f'psnr:{psnr}\n')
	class_utils.append_to_file(log_path + "generator.txt", f'ssim:{ssim}\n')

# Vypočíta hodnoty porovnávacích metrík pre zvacseny dataset s originalym nezmensenym datasetom,
# ktoré následne zapíše do súboru
def test_generator_no_acc(generator,dataset,log_path='log/methods_test/tests/'):
	print(f'Starting Generator Test')
	total_mse = 0
	total_ssim = 0
	total_psnr = 0
	total_euclid = 0
	total_manhattan = 0
	batches_done = 0

	for batch in dataset:
		orig_batch, downsampled = batch[:]
		batch_size = orig_batch.shape[0]
		generated = generator.output(downsampled, is_training=False)
		total_manhattan += metrics.manhattan_distance(orig_batch, generated)
		total_mse += metrics.mean_square_error(orig_batch, generated)
		total_ssim += metrics.SSIM(orig_batch, generated)
		total_psnr += metrics.PSNR(orig_batch, generated)
		total_euclid += metrics.euclid_distance(orig_batch, generated)
		batches_done += 1

	print(batches_done,batch_size)
	mse = total_mse / (batches_done * batch_size)
	euclid = total_euclid / (batches_done * batch_size)
	psnr = total_psnr / (batches_done * batch_size)
	ssim = total_ssim / (batches_done * batch_size)
	print('Generator  tested')
	class_utils.append_to_file(log_path + f"generator.txt", f'mse:{mse}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'euclid:{euclid}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'psnr:{psnr}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'ssim:{ssim}\n')

# Vypočíta hodnoty porovnávacích metrík a aj výsledok klasifikácie pre zvacseny dataset generatorom ,
# ktoré následne zapíše do súboru
def test_generator_v2(generator,classifier,dataset,log_path='log/methods_test/tests/'):
	print(f'Starting Generator Test')
	total_mse = 0
	total_ssim = 0
	total_psnr = 0
	total_euclid = 0
	total_manhattan = 0
	batches_done = 0
	total_correct_predictions = 0
	total_wrong_predictions = 0

	for batch in dataset:
		orig_batch, downsampled, label = batch[:]
		batch_size = orig_batch.shape[0]
		generated = generator.output(downsampled, is_training=False)
		total_manhattan += metrics.manhattan_distance(orig_batch, generated)
		total_mse += metrics.mean_square_error(orig_batch, generated)
		total_ssim += metrics.SSIM(orig_batch, generated)
		total_psnr += metrics.PSNR(orig_batch, generated)
		total_euclid += metrics.euclid_distance(orig_batch, generated)
		predicted = classifier.output(generated, training=False)
		real_classes = tf.keras.backend.argmax(label, axis=1)
		predicted_classes = tf.keras.backend.argmax(predicted, axis=1)
		total_correct_predictions += tf.keras.backend.sum(
			tf.keras.backend.cast(tf.keras.backend.equal(real_classes, predicted_classes), tf.float32))
		total_wrong_predictions += tf.keras.backend.sum(
			tf.cast(tf.keras.backend.not_equal(real_classes, predicted_classes), tf.float32))
		batches_done += 1

	accuracy = total_correct_predictions / (total_correct_predictions + total_wrong_predictions)
	mse = total_mse / (batches_done * batch_size)
	euclid = total_euclid / (batches_done * batch_size)
	psnr = total_psnr / (batches_done * batch_size)
	ssim = total_ssim / (batches_done * batch_size)
	print('Generator  tested')
	class_utils.append_to_file(log_path + f"generator.txt", f'accuracy:{accuracy}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'mse:{mse}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'euclid:{euclid}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'psnr:{psnr}\n')
	class_utils.append_to_file(log_path + f"generator.txt", f'ssim:{ssim}\n')

# Vypočíta hodnoty porovnávacích metrík pre zvacseny dataset zväčšovacou metódou
# s originalym nezmensenym datasetom, ktoré následne zapíše do súboru
def test_upscale_method_no_acc(method, dataset, methodName, log_path='log/methods_test/tests/'):
	print(f'Starting {methodName} Test')
	total_mse = 0
	total_ssim = 0
	total_psnr = 0
	total_euclid = 0
	total_manhattan = 0
	batches_done = 0
	total_samples = 0
	for batch in dataset:
		orig_batch, downsampled = batch[:]
		batch_size = orig_batch.shape[0]
		generated = img.upsample(downsampled, (orig_batch.shape[1], orig_batch.shape[2]), method)
		total_manhattan += metrics.manhattan_distance(orig_batch, generated)
		total_mse += metrics.mean_square_error(orig_batch, generated)
		total_ssim += metrics.SSIM(orig_batch, generated)
		total_psnr += metrics.PSNR(orig_batch, generated)
		total_euclid += metrics.euclid_distance(orig_batch, generated)
		total_samples += batch_size


	mse = total_mse / total_samples
	euclid = total_euclid / total_samples
	psnr = total_psnr / total_samples
	ssim = total_ssim /total_samples
	print('Method  tested')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'mse:{mse}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'euclid:{euclid}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'psnr:{psnr}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'ssim:{ssim}\n')
# Vypočíta hodnoty porovnávacích metrík a aj hodnotu klasifikácie pre zvacseny dataset  zväčšovacou metódou s originalym nezmensenym datasetom,
# ktoré následne zapíše do súboru
def test_upscale_method(method,classifier,dataset,methodName,log_path='log/methods_test/tests/'):
	print(f'Starting {methodName} Test')
	total_mse = 0
	total_ssim = 0
	total_psnr = 0
	total_euclid = 0
	total_manhattan = 0
	batches_done = 0
	total_correct_predictions = 0
	total_wrong_predictions = 0

	for batch in dataset:
		orig_batch, downsampled, label = batch[:]
		batch_size = orig_batch.shape[0]
		generated = img.upsample(downsampled,(orig_batch.shape[1],orig_batch.shape[2]),method)
		total_manhattan += metrics.manhattan_distance(orig_batch, generated)
		total_mse += metrics.mean_square_error(orig_batch, generated)
		total_ssim += metrics.SSIM(orig_batch, generated)
		total_psnr += metrics.PSNR(orig_batch, generated)
		total_euclid += metrics.euclid_distance(orig_batch, generated)
		predicted = classifier.output(generated, training=False)
		real_classes = tf.keras.backend.argmax(label, axis=1)
		predicted_classes = tf.keras.backend.argmax(predicted, axis=1)
		total_correct_predictions += tf.keras.backend.sum(
			tf.keras.backend.cast(tf.keras.backend.equal(real_classes, predicted_classes), tf.float32))
		total_wrong_predictions += tf.keras.backend.sum(
			tf.cast(tf.keras.backend.not_equal(real_classes, predicted_classes), tf.float32))
		batches_done += 1

	accuracy = total_correct_predictions / (total_correct_predictions + total_wrong_predictions)
	mse = total_mse / (batches_done * batch_size)
	euclid = total_euclid / (batches_done * batch_size)
	psnr = total_psnr / (batches_done * batch_size)
	ssim = total_ssim / (batches_done * batch_size)
	print('Method  tested')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'accuracy:{accuracy}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'mse:{mse}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'euclid:{euclid}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'psnr:{psnr}\n')
	class_utils.append_to_file(log_path + f"{methodName}.txt", f'ssim:{ssim}\n')


