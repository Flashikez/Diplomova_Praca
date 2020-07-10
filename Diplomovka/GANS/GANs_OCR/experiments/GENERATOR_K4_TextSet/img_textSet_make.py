
import tensorflow as tf

import Diplomovka.GANs.Utils.dataset_utils as datasets
import Diplomovka.img_utils as img
from matplotlib import pyplot as plt

train_path = '../../datasets/Arial_Black_Regular_12_noborder/train'
test_path = '../datasets//Arial_Black_Regular_12_noborder/test'





batch_size = 50

def map_fn(imgs,labels):
	downsampled = img.downsample(imgs,(56,56))
	return imgs,downsampled,labels

train_size = 30000
test_size = 10000
train_set = datasets.make_dataset(train_path,batch_size).map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

for batch in train_set:
	batchs = batch
	break

imgs = batchs[0]


fig, ax = plt.subplots(10, 5,constrained_layout=True)
for i in range(10):
	for j in range(5):
		ax[i, j].axis("off")
		ax[i, j].imshow(imgs[i*j,:,:,0],cmap="gray")

plt.show()
