
import Diplomovka.img_utils as img
import tensorflow as tf
import Diplomovka.GANs.Utils.metrics_calculator as calc
import Diplomovka.GANs.Utils.dataset_utils as load

test_path = '../../datasets/Arial_Black_Regular_12_noborder_more/test'
test_size = 10000
batch_size = 16

def map_fn(imgs,labels):
	downsampled = img.downsample(imgs, (56, 56))
	# Tu zväčši imgs nejakou metódou
	# upsampled = generator.output(downsampled,is_training=False)
	# upsampled = img.upsample(downsampled,(112,112),tf.image.ResizeMethod.LANCZOS5)
	return imgs,downsampled

test_datasets_methods =load.make_dataset(test_path,batch_size).map(map_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)


calc.test_upscale_method_no_acc(tf.image.ResizeMethod.NEAREST_NEIGHBOR,test_datasets_methods,'NEAREST')
calc.test_upscale_method_no_acc(tf.image.ResizeMethod.BILINEAR,test_datasets_methods,'BILINEAR')
calc.test_upscale_method_no_acc(tf.image.ResizeMethod.BICUBIC,test_datasets_methods,'BICUBIC')
calc.test_upscale_method_no_acc(tf.image.ResizeMethod.LANCZOS3 ,test_datasets_methods,'LANCZOS3')
calc.test_upscale_method_no_acc(tf.image.ResizeMethod.LANCZOS5 ,test_datasets_methods,'LANCZOS5')



