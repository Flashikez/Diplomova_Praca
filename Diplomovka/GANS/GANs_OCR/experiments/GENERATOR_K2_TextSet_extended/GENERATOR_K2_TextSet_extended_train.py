import tensorflow as tf
from Diplomovka.GANs.architectures.Generators.GENERATOR_K2 import GENERATOR_K2
from Diplomovka.GANs.architectures.Discriminators.DISCRIM import DISCRIM
from Diplomovka.GANs.GANs_OCR.trainers.GAN_Best_SSIM_SAVEALL_epochs_TEXTSet_Trainer import GAN_Best_SSIM_TEXTSet_SAVEALL_epochs_Trainer as Trainer
import Diplomovka.GANs.Utils.dataset_utils as datasets
import Diplomovka.img_utils as img


train_path = '../../Arial_Black_Regular_12_noborder_more/train'
test_path = '../../Arial_Black_Regular_12_noborder_more/test'




batch_size = 8

def map_fn(imgs,labels):
	downsampled = img.downsample(imgs,(56,56))
	return imgs,downsampled,labels

train_size = 60000
test_size = 10000
train_set = datasets.make_dataset(train_path,batch_size).map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_set =  datasets.make_dataset(test_path,batch_size).map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# # -------------------------------------

# # -------------------------------------


discriminator = DISCRIM([112,112,1])
generator = GENERATOR_K2([56,56,1])


# generator.loadWeights('models/generator/generator/56to112_10EPOCHS/generator_56to112_30epochs')
# trainer.mse_test(test_set.__iter__(),generator,'img_figures_56to112/','1')
# maker.make_dataset_from_generator(generator,test_set,'generated_datasets/56to112/')



trainer = Trainer(train_set,test_set,generator,discriminator,batch_size,train_size,test_size)
trainer.start('pretrained_models/generator/','pretrained_models/discirimnator/',plot_title="Tréning GENERATOR_K2 na rozšírenom datasete TextSet",epochs=20)