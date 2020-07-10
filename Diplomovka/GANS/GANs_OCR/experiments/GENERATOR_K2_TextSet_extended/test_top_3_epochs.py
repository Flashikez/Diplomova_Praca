from Diplomovka.GANs.architectures.Generators.GENERATOR_K2 import GENERATOR_K2
import Diplomovka.img_utils as img
import Diplomovka.GANs.Utils.dataset_utils as load
import Diplomovka.GANs.GANs_OCR.tesseract_tests as tess_test
import tensorflow as tf



# Otestuje všetky zväčšovacie metódy a TOP3 epochy generátora, na OCR so zaznamenaním počtu správne/nesprávne rozpoznaných znakov
import numpy as np


def decode_log_file_metric(path):
	with open(path, "r") as f1:
		lines = f1.readlines()
		lines = lines[1:]
		x, y,y_av = [list(d) for d in zip(*[[i for i in c.split(':')] for c in lines])]
		x = list(map(int, x))
		y_av = list(map(float, y_av))
		return np.array(x), np.array(y_av)

x,y = decode_log_file_metric('log/SSIM.txt')
print(y)
# indexy top 3 epochov
idx = (-y).argsort()[:3]
print(idx)

top_epoch,second_epoch,third_epoch = idx[:]



test_path = '../../datasets/Arial_Black_Regular_12_noborder_more/test'
test_size = 10000
batch_size = 16


#
#
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS5)
# 	return upsampled,labels
#
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # num_of_batches_to_plot = 6
# # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'LANCZOS5')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS3)
# 	return upsampled,labels
#
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # num_of_batches_to_plot = 6
# # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'LANCZOS3')
#
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.BILINEAR)
# 	return upsampled,labels
#
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # num_of_batches_to_plot = 6
# # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'BILINEAR')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.BICUBIC)
# 	return upsampled,labels
#
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # num_of_batches_to_plot = 6
# # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'BICUBIC')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 	return upsampled,labels
#
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # num_of_batches_to_plot = 6
# # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'NEAREST')
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	return downsampled,labels
#
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # num_of_batches_to_plot = 6
# # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'LOW_RES')
#
#
# generator_top = GENERATOR_K2([56,56,1])
# generator_top.loadWeights(f'pretrained_models/generator/{top_epoch}/generator')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = generator_top.output(downsampled,is_training=False)
#
# 	return upsampled,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# #
# # # num_of_batches_to_plot = 6
# # # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'GENERATOR_BEST')
#
# del generator_top
# generator_second = GENERATOR_K2([56,56,1])
# generator_second.loadWeights(f'pretrained_models/generator/{second_epoch}/generator')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
#
# 	upsampled = generator_second.output(downsampled,is_training=False)
#
# 	return upsampled,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
#
# # # num_of_batches_to_plot = 6
# # # progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
# # test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'GENERATOR_SECOND')

# del generator_second
generator_third = GENERATOR_K2([56,56,1])
generator_third.loadWeights(f'pretrained_models/generator/{third_epoch}/generator')

def map_fn(imgs,labels):
	downsampled = img.downsample(imgs, (56, 56))

	upsampled = generator_third.output(downsampled,is_training=False)

	return upsampled,labels

# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)

# num_of_batches_to_plot = 6
# progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
tess_test.accuracy_on_dataset_log_alphabet(test_set,test_size,batch_size,'GENERATOR_THIRD')