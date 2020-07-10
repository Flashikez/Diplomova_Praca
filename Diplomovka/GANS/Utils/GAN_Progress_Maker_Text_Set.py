import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Trieda vytvára obrázok, na ktorom sú batche obrázkov vkladané do riadkov, používaná pri datasete Text_Set
class Progress_Maker_Text_Set():


	def __init__(self,title,num_of_images_per_row=8,rows=12+3):
		self.index = 0
		self.rows = rows
		self.columns =  num_of_images_per_row

		self.fig = plt.figure(figsize=(self.columns + 1, self.rows + 1))

		self.gs = gridspec.GridSpec(self.rows, self.columns,
							   wspace=0.0, hspace=0.0,
							   top=1. - 0.5 / (self.rows + 1), bottom=0.5 / (self.rows + 1),
							   left=0.5 / (self.columns + 1), right=1 - 0.5 / (self.columns + 1))

		self.fig.suptitle(title ,y=0.99)

	def save_progress(self,path,name):
		plt.savefig((path + '{}.png').format(name))

	def plot_img(self,img,axis):
		# Unnormalizeň
		import tensorflow as tf
		img = img[:, :, 0] * 127.5 + 127.5
		img = tf.round(img)
		img = tf.clip_by_value(img,0,255)

		# print(img.numpy())
		# input()
		img = np.uint8(img)

		# print(img.numpy())
		# input()
		axis.imshow(img,cmap='gray')

	def add_batch(self,batch,title):
		label = True
		for i in range(self.columns):
			ax = plt.subplot(self.gs[self.index])
			self.index+=1
			# ax.set_title("cau")
			self.plot_img(batch[i],ax)

			# ax.set_xticklabels("lol")

			ax.xaxis.set_visible(False)
			if(not label):
				ax.yaxis.set_visible(False)
			if label:
				ax.set_ylabel(title,fontsize=8)
				label = False

	def show(self):
		plt.show()






















