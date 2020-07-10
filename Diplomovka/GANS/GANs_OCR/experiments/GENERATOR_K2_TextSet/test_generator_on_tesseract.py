import random
import tensorflow as tf
import Diplomovka.img_utils as img
import Diplomovka.GANs.Utils.dataset_utils as load
import Diplomovka.GANs.GANs_OCR.tesseract_tests as tess_test
from Diplomovka.GANs.Utils.GAN_Progress_Maker import Progress_Maker
from Diplomovka.GANs.architectures.Generators.GENERATOR_K2 import GENERATOR_K2

test_path = '../../datasets/Arial_Black_Regular_12_noborder/test'
test_size = 5000
batch_size = 32

generator = GENERATOR_K2([56,56,1])
generator.loadWeights('pretrained_models/generator/generator')



def map_fn(imgs,labels):
	downsampled = img.downsample(imgs, (56, 56))
	# Tu zväčši imgs nejakou metódou
	# upsampled = generator.output(downsampled,is_training=False)
	# upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS5)
	return downsampled,labels

# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)

# num_of_batches_to_plot = 6
# progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_low_res')
# count = 0
# skip_first= True
# for batch in test_set:
# 	origs,downsampled,restored = batch[:]
# 	if skip_first:
# 		skip_first = False
# 		continue
# 	progress_maker.add_batch(origs,'Originál')
# 	progress_maker.add_batch(downsampled, 'Zmenšené')
# 	progress_maker.add_batch(restored, 'Zväčšené')
# 	count +=1
# 	if count == num_of_batches_to_plot:
# 		break
# progress_maker.save_progress('examples/','examples_BICUBIC-ss')



#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	# Tu zväčši imgs nejakou metódou
# 	# imgs = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS5)
# 	return imgs,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_original')
# #
# #
# #
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	# Tu zväčši imgs nejakou metódou
# 	imgs = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 	return imgs,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_NEAREST')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	# Tu zväčši imgs nejakou metódou
# 	imgs = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS3)
# 	return imgs,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_LANCZOS3')
#
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	# Tu zväčši imgs nejakou metódou
# 	imgs = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS5)
# 	return imgs,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_LANCZOS5')
# #
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	# Tu zväčši imgs nejakou metódou
# 	imgs = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.BILINEAR)
# 	return imgs,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_BILINEAR')
# #
# #
# def map_fn(imgs,labels):
# 	downsampled = img.downsample(imgs, (56, 56))
# 	# Tu zväčši imgs nejakou metódou
# 	imgs = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.BICUBIC)
# 	return imgs,labels
#
# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
# tess_test.accuracy_on_dataset(test_set,test_size,batch_size,'Accuracy_BICUBIC')
#
#

