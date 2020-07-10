
import tensorflow as tf


# Cim blizsia hodnote 1, tym su viac podobne
def SSIM(x1,x2,unnormalize_inputs=True):
	if unnormalize_inputs:
		x1 = unnormalize(x1)
		x2 = unnormalize(x2)

	return tf.reduce_sum(tf.image.ssim(x1,x2,max_val=255))
# Cim vacsia tym sa viac podobaju
def PSNR(x1,x2,unnormalize_inputs=True):
	if unnormalize_inputs:
		x1 = unnormalize(x1)
		x2 = unnormalize(x2)

	return  tf.reduce_sum(tf.image.psnr(x1,x2,max_val=255))


# Cim mensia hodnota, tym sa viac x1,x2 podobaju
def mean_square_error(x1,x2,unnormalize_inputs=True):
	if unnormalize_inputs:
		x1 = unnormalize(x1)
		x2 = unnormalize(x2)

	return  tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(x1,x2)),axis=(1,2)))


# Cim mensia hodnota, tym sa viac x1,x2 podobaju
def euclid_distance(x1,x2,unnormalize_inputs=True):
	if unnormalize_inputs:
		x1 = unnormalize(x1)
		x2 = unnormalize(x2)


	return  tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1,x2)),axis=(1,2))))



def normalize(imgs):
	return (imgs - 127.5) / 127.5

def unnormalize(imgs):
	return (imgs * 127.5)+127.5

def upsample(imgs,size,method):
	return tf.image.resize(imgs,size,method=method)

