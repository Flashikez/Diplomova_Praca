
import tensorflow as tf
import Diplomovka.img_utils as img
from Diplomovka.GANs.architectures.Generators.GENERATOR_K4 import GENERATOR_K4
from Diplomovka.GANs.Utils.GAN_Progress_Maker import Progress_Maker
import Diplomovka.GANs.Utils.dataset_utils as load


test_path = '../../datasets/Arial_Black_Regular_12_noborder/test'
test_size = 5000
batch_size = 8

generator = GENERATOR_K4([28,28,1])
generator.loadWeights('pretrained_models/generator/generator')








def map_fn(imgs,labels):
	downsampled = img.downsample(imgs, (28, 28))
	# Tu zväčši imgs nejakou metódou
	# upsampled = generator.output(downsampled,is_training=False)
	# upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS5)
	return imgs,downsampled

# test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)

num_of_batches_to_plot = 6
# progress_maker = Progress_Maker('Príklad zväčšených obrázkov FMNIST generátorom GENERATOR_K4 trénovanom na percepčnej chybe',8,num_of_batches_to_plot*3)
test_set =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)

num_of_batches_to_plot = 3
progress_maker = Progress_Maker('Príklad zväčšených obrázkov Test_Set generátorom #1 GENERATOR_K2 a metódou LANCZOS5',batch_size,num_of_batches_to_plot*4)


count = 0
skip_first= True
for batch in test_set:
	origs,downsampled = batch[:]
	if skip_first:
		skip_first = False
		continue

	progress_maker.add_batch(origs, 'Originál')
	progress_maker.add_batch(downsampled, 'Zmenšené')
	generator_imgs = generator.output(downsampled,is_training=False)
	progress_maker.add_batch(generator_imgs, 'generátor')
	lanczos3 = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS3)
	progress_maker.add_batch(lanczos3,'LANCZOS5',clip_batch=True)
	# restored = numpy_data_bw(restored)
	# restored =

	count +=1
	if count == num_of_batches_to_plot:
		break
progress_maker.save_progress('examples/','examples_GEN_LANCZ5')


