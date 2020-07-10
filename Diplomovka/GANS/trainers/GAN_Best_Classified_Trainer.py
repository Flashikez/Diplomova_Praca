import tensorflow as tf
import os
import shutil
import Diplomovka.Classifiers.Classifier_utils as class_utils
import Diplomovka.GANs.Utils.metrics as metrics
from Diplomovka.GANs.Utils.GAN_Progress_Maker import Progress_Maker

# Tréner GANu metódou najlepšej klasifikácie na klasifikátore classifier, s klasickou chybovou funkciou generátora
class GAN_classic_loss_trainer():
	def __init__(self,real_dataset,generator_inputs_dataset,test_generator_inputs_dataset,test_dataset_original,accuracy_dataset,generator,discriminator,classifier,batch_size):
		self.real_dataset = real_dataset

		self.generator_input_dataset = generator_inputs_dataset

		self.test_dataset_gen_inputs = test_generator_inputs_dataset
		self.test_dataset_real = test_dataset_original
		self.accuracy_dataset = accuracy_dataset


		self.classifier = classifier
		self.generator = generator
		self.discriminator = discriminator
		self.batch_size = batch_size
		self.train_steps_done = 0

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


	def start(self,gen_save_path,discr_save_path,generator_save_name = "generator",discrim_save_name="discriminator",epochs=12,save_best_classified_model=True,plot_batch_after_epoch=True,plot_batch_size=8,plot_path='plot/',log_losses=True,log_metrics=True,log_accuracy = True, log_path="log/",plot_title='Tréning GANu'):
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)



		if plot_batch_after_epoch:
			for test_batch,orig_batch in zip(self.test_dataset_gen_inputs,self.test_dataset_real):
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
		if log_accuracy:
			print('Calculating accuracy of generated test set on classifier before training')
			accuracy = self.generator_accuracy_test(self.generator,self.classifier,self.accuracy_dataset)
			self.log_accuracy(0,accuracy,log_path)

		if log_metrics:
			print('Calculating metrics of generated test vs original')
			metrics_dict = self.calculate_metrics(self.generator,self.test_dataset_gen_inputs,self.test_dataset_real)
			self.log_metrics(0,metrics_dict,log_path)



		bestAccuracy = 0
		for epoch in range(1,epochs+1):


			batches_done = 0
			for batch_real,batch_gen_inputs in zip(self.real_dataset,self.generator_input_dataset):
				#print(batch_gen_inputs.get_shape())
				#print(batch_real.get_shape())
				gen_loss,discrim_loss = self._train_step(batch_real,batch_gen_inputs)

				if self.train_steps_done % 10 ==0:
					if log_losses:
						self.log_losses(self.train_steps_done, gen_loss, discrim_loss, log_path)
				self.train_steps_done += 1
				batches_done+=1
				if batches_done % 10 == 0:
					print('Epoch: ',epoch,' batches done: ',batches_done,'out of ',60000//self.batch_size)

			if plot_batch_after_epoch:
				generated_batch = self.generator.output(plot_batch_gen_inputs,is_training=False)
				progress_maker.add_batch(generated_batch, f'Po {epoch} epochu')
				progress_maker.save_progress(plot_path, f'after_{epoch}')
			if log_metrics:
				print('Calculating metrics of generated test vs original')
				metrics_dict = self.calculate_metrics(self.generator, self.test_dataset_gen_inputs,
													  self.test_dataset_real)
				self.log_metrics(epoch, metrics_dict, log_path)

			# if log_losses:
			# 	print('Calculating generator and discriminator losses on test set')
			# 	gen_loss, discr_loss = gan_utils.classic_losses_on_dataset(self.generator, self.discriminator,
			# 															   self.test_dataset_real,self.test_dataset_gen_inputs)
			# 	self.log_losses(epoch, gen_loss, discr_loss, log_path)


			print('Epoch:',epoch ,' done')
			print('Calculating accuracy of classifier on generated test dataset from generator')
			accuracy = self.generator_accuracy_test(self.generator,self.classifier,self.accuracy_dataset)
			if log_accuracy:
				self.log_accuracy(epoch, accuracy, log_path)

			print(f'accuracy: {accuracy}')
			if accuracy > bestAccuracy and save_best_classified_model is True:
				print(f'Found best accuracy epoch {epoch}, accuracy {accuracy}')
				print('Saving models')
				self.save_models(self.generator, self.discriminator, gen_save_path + generator_save_name,
								 discr_save_path + discrim_save_name)
				bestAccuracy = accuracy

		print('Training done')
		if not save_best_classified_model:
			print('Saving models')
			self.save_models(self.generator,self.discriminator,gen_save_path+generator_save_name,
							 discr_save_path+discrim_save_name)


	def save_models(self,generator,discriminator,generator_save_path,discriminator_save_path):
		shutil.rmtree(generator_save_path, ignore_errors=True)
		shutil.rmtree(discriminator_save_path, ignore_errors=True)
		discriminator.saveWeights(discriminator_save_path)
		generator.saveWeights(generator_save_path)

	@staticmethod
	def generator_accuracy_test(generator,classifier,test_dataset):
		def generated_test_f():
			for img, label in test_dataset:
				generated = generator.output(img, is_training=False)
				# generated = io.unnormalize(generated)
				yield generated, label

		generated_dataset = tf.data.Dataset.from_generator(generated_test_f,(tf.float32,tf.int64))
		accuracy = class_utils.accuracy_on_dataset(classifier,generated_dataset)
		print('Ended accuracy generation')
		return accuracy

	def log_losses(self,epoch,gen_loss,discr_loss,log_path):
		if epoch == 0:
			class_utils.append_to_file(f"{log_path}trainlog_gen_loss.txt", "epoch:loss\n")
			class_utils.append_to_file(f"{log_path}trainlog_discr_loss.txt", "epoch:loss\n")

		class_utils.append_to_file(f"{log_path}trainlog_gen_loss.txt", f"{epoch}:{gen_loss}\n")
		class_utils.append_to_file(f"{log_path}trainlog_discr_loss.txt", f"{epoch}:{discr_loss}\n")

	def log_accuracy(self,epoch,acc,log_path):
		if epoch == 0:
			class_utils.append_to_file(f"{log_path}trainlog_class_accuracy.txt", "epoch:loss\n")

		class_utils.append_to_file(f"{log_path}trainlog_class_accuracy.txt", f"{epoch}:{acc}\n")

	def log_metrics(self,epoch,metrics_dict,log_path):
		for k,v in metrics_dict.items():
			if epoch == 0:
				class_utils.append_to_file(f"{log_path}{k}.txt", "epoch:metric_total:metric_average\n")

			class_utils.append_to_file(f"{log_path}{k}.txt", f"{epoch}:{v[0]}:{v[1]}\n")




	def calculate_metrics(self,generator,generator_inputs_dataset,original_dataset):
		total_mse = 0
		total_ssim = 0
		total_psnr = 0
		total_euclid = 0
		total_manhattan = 0
		batches_done = 0

		for test_batch, orig_batch in zip(generator_inputs_dataset, original_dataset):
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


