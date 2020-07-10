import tensorflow as tf
import Diplomovka.img_utils as img
from Diplomovka.GANs.architectures.Generators.GENERATOR_K2 import GENERATOR_K2
from Diplomovka.GANs.Utils.GAN_Progress_Maker_Text_Set import Progress_Maker_Text_Set
import Diplomovka.GANs.Utils.dataset_utils as load



# Vytovrí príklady zväčšenia obrázkov pomocou GANu alebo zväčšenia obrázkov pomocou inej metódy
test_path = '../../datasets/Arial_Black_Regular_12_noborder/test'
test_size = 5000
batch_size = 6

generator = GENERATOR_K2([56,56,1])
generator.loadWeights('pretrained_models/generator/generator')

def map_fn(imgs,labels):
	downsampled = img.downsample(imgs, (56, 56))
	return imgs,downsampled


num_of_batches_to_plot = 6
test_set =load.make_dataset(test_path,batch_size,seed=1234).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
generator = GENERATOR_K2([56,56,1])
generator.loadWeights('pretrained_models/generator/generator')


num_of_batches_to_plot = 6
progress_maker = Progress_Maker_Text_Set('Príklad zväčšených obrázkov Test_Set interpoláciou LANCZOS,',batch_size,num_of_batches_to_plot*3)


count = 0
skip_first= True
first_two = True
for batch in test_set:

	origs,downsampled = batch[:]
	if skip_first:
		skip_first = False
		continue
	upsampled = generator.output(downsampled,is_training=False)
	upsampled_lan = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS3 )
	# upsampled_gen = generator.output(downsampled,is_training=False)

	progress_maker.add_batch(origs,'Originál')
	progress_maker.add_batch(downsampled, 'Zmenšené')
	progress_maker.add_batch(upsampled_lan, 'Zväčšené')
	count +=1
	if count == num_of_batches_to_plot:
		break
progress_maker.save_progress('examples/','examples_LANCZOS3')


