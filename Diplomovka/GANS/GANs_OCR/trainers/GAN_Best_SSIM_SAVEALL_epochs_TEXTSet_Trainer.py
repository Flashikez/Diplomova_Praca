import tensorflow as tf
import os


import Diplomovka.Classifiers.Classifier_utils as class_utils

import Diplomovka.GANs.Utils.metrics as metrics
from Diplomovka.GANs.Utils.GAN_Progress_Maker import Progress_Maker
# Trénuje GAN, ukladá modely po všetkých epochách, vypisuje priebežne SSIM index
class GAN_Best_SSIM_TEXTSet_SAVEALL_epochs_Trainer():
	def __init__(self,train_dataset,test_dataset,generator,discriminator,batch_size,train_size,test_size):
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.generator = generator
		self.discriminator = discriminator
		self.batch_size = batch_size
		self.train_steps_done = 0
		self.train_size = train_size

	@tf.function
	# Realizuje tréningový krok trénovania GANov
	# real_images - batch pochádzajúci z pôvodnej trénovacej množiny
	# gen_input - batch vstupných dát generátora, teda v prípade zväčšovania rozlíšenia batch zmenšenej tréningovej množiny
	def _train_step(self,real_images,gen_input):
		with tf.GradientTape() as generator_tape , tf.GradientTape() as discrim_tape:
			# Výstup generátora. Hodnota G(z)
			gen_imgs = self.generator.output(gen_input,is_training=True)
			# Výstup diskrimátora ak na vstupe bol batch z pôvodnej trénovacej množiny
			# Hodnota D(x)
			real_output = self.discriminator.output(real_images,is_training=True)
			# Výstup diskrimátora ak na vstupe bol batch generovaný generátorom
			# Hodnota D(G(z))
			fake_output = self.discriminator.output(gen_imgs,is_training=True)
			# Hodnota chybovej funkcie generátora
			gen_loss = self.generator.loss(fake_output)
			# Hodnota chybovej funkcie diskriminátora
			discrim_loss = self.discriminator.loss(real_output,fake_output)


		# Úprava váh generátora a diskriminátora
		self.generator.calc_apply_gradients(generator_tape,gen_loss)
		self.discriminator.calc_apply_gradients(discrim_tape,discrim_loss)
		return gen_loss,discrim_loss


	def start(self,gen_save_path,discr_save_path,generator_save_name = "generator",discrim_save_name="discriminator",epochs=12,save_best_classified_model=True,plot_batch_after_epoch=True,plot_batch_size=8,plot_path='plot/',log_losses=True,log_metrics=True, log_path="log/",plot_title='Tréning GANu'):
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)



		if plot_batch_after_epoch:
			for batch in self.test_dataset:
				orig_batch,test_batch,_ = batch[:]
				plot_batch_gen_inputs = test_batch[:plot_batch_size]
				plot_batch_real = orig_batch[:plot_batch_size]
				progress_maker = Progress_Maker(plot_title, plot_batch_size, epochs+3)
				progress_maker.add_batch(plot_batch_real, 'Originál')
				progress_maker.add_batch(plot_batch_gen_inputs, 'Zmenšené')
				generated_batch = self.generator.output(plot_batch_gen_inputs, is_training=False)
				progress_maker.add_batch(generated_batch, f'Pred tréningom')
				progress_maker.save_progress(plot_path,'before_training')
				break

		# if log_losses:
		# 	print('Calculating generator and discriminator losses on test set before training')
		# 	gen_loss, discr_loss = gan_utils.classic_losses_on_dataset(self.generator, self.discriminator,
		# 															   self.real_dataset,self.generator_input_dataset)
		# 	self.log_losses(0,gen_loss,discr_loss,log_path)


		if log_metrics:
			print('Calculating metrics of generated test vs original')
			metrics_dict = self.calculate_metrics(self.generator,self.test_dataset)
			self.log_metrics(0,metrics_dict,log_path)

			self.save_models(self.generator, self.discriminator, f'{gen_save_path}/{0}/{generator_save_name}',
							 f'{discr_save_path}/{0}/{discrim_save_name}')


		for epoch in range(1,epochs+1):


			batches_done = 0
			for batch in self.train_dataset:
				batch_real,batch_gen_inputs,_ = batch[:]
				#print(batch_gen_inputs.get_shape())
				#print(batch_real.get_shape())
				gen_loss,discrim_loss = self._train_step(batch_real,batch_gen_inputs)

				if self.train_steps_done % 10 ==0:
					if log_losses:
						self.log_losses(self.train_steps_done, gen_loss, discrim_loss, log_path)
				self.train_steps_done += 1
				batches_done+=1
				if batches_done % 10 == 0:
					print('Epoch: ',epoch,' batches done: ',batches_done,'out of ',self.train_size//self.batch_size)

			print('Epoch:',epoch ,' done')
			if plot_batch_after_epoch:
				generated_batch = self.generator.output(plot_batch_gen_inputs,is_training=False)
				progress_maker.add_batch(generated_batch, f'Po {epoch} epochu')
				progress_maker.save_progress(plot_path, f'after_{epoch}')

			print('Calculating metrics of generated test vs original')

			metrics_dict = self.calculate_metrics(self.generator, self.test_dataset)
			if log_metrics:
				self.log_metrics(epoch, metrics_dict, log_path)
			metric_value = metrics_dict['SSIM'][1]

			print(f'Metric Value: {metric_value}')
			print('Saving models')
			self.save_models(self.generator, self.discriminator, f'{gen_save_path}/{epoch}/{generator_save_name}',
							 f'{discr_save_path}/{epoch}/{discrim_save_name}')



	def save_models(self,generator,discriminator,generator_save_path,discriminator_save_path):
		discriminator.saveWeights(discriminator_save_path)
		generator.saveWeights(generator_save_path)


	def log_losses(self,epoch,gen_loss,discr_loss,log_path):
		if epoch == 0:
			class_utils.append_to_file(f"{log_path}trainlog_gen_loss.txt", "epoch:loss\n")
			class_utils.append_to_file(f"{log_path}trainlog_discr_loss.txt", "epoch:loss\n")

		class_utils.append_to_file(f"{log_path}trainlog_gen_loss.txt", f"{epoch}:{gen_loss}\n")
		class_utils.append_to_file(f"{log_path}trainlog_discr_loss.txt", f"{epoch}:{discr_loss}\n")


	def log_metrics(self,epoch,metrics_dict,log_path):
		for k,v in metrics_dict.items():
			if epoch == 0:
				class_utils.append_to_file(f"{log_path}{k}.txt", "epoch:metric_total:metric_average\n")

			class_utils.append_to_file(f"{log_path}{k}.txt", f"{epoch}:{v[0]}:{v[1]}\n")




	def calculate_metrics(self,generator,dataset):
		total_mse = 0
		total_ssim = 0
		total_psnr = 0
		total_euclid = 0
		total_manhattan = 0
		batches_done = 0

		for batch in dataset:
			orig_batch,test_batch,_ = batch[:]
			batch_size = test_batch.shape[0]
			generated = generator.output(test_batch,is_training=False)
			total_manhattan += metrics.manhattan_distance(orig_batch,generated)
			total_mse += metrics.mean_square_error(orig_batch,generated)
			total_ssim += metrics.SSIM(orig_batch,generated)
			total_psnr += metrics.PSNR(orig_batch,generated)
			total_euclid += metrics.euclid_distance(orig_batch,generated)
			batches_done += 1

		total_samples = batch_size*batches_done
		return {'MSE':(total_mse,total_mse/total_samples),'EUCL':(total_euclid,total_euclid/total_samples),'MANH':(total_manhattan,total_manhattan/total_samples),'SSIM':(total_ssim,total_ssim/total_samples),'PSNR':(total_psnr,total_psnr/total_samples)}


