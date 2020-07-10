import tensorflow as tf
import os
import shutil
import Diplomovka.Classifiers.Classifier_utils as utils

# Tréner klasifikátora classifier
class Classifier_Trainer():
	def __init__(self,train_dataset,test_dataset,classifier,batch_size=32,train_data_size = 60000):
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.test_dataset_iterator = self.test_dataset.__iter__()
		self.classifier = classifier
		self.batch_size = batch_size
		self.train_data_size =train_data_size
		self.training_steps_done = 0

	@tf.function
	def _train_step(self,inputs,labels):
		with tf.GradientTape() as classifier_tape:

			predicted = self.classifier.output(inputs,training=True)
			loss = self.classifier.loss(labels,predicted)

		self.classifier.calc_apply_gradients(classifier_tape,loss)
		return loss
	# Odštartuje tréningový proces
	def start(self,classifier_save_path,save_name='classifier',epochs=16,save_best_classified_model = True,log_loss = True,log_test_dataset_accuracy_after_epoch=True,log_path = 'log/'):
		self.training_steps_done = 0
		bestAccuracy = 0

		if log_test_dataset_accuracy_after_epoch:
			print(f'Calculating accuracy before any training')
			acc = utils.accuracy_on_dataset(self.classifier, self.test_dataset)
			print(f'Accuracy: {acc}')
			self.log_accuracy(0,acc,log_path)

		print('Training started')
		for epoch in range(1,epochs+1):

			batches_done = 0
			for imgs,labels in self.train_dataset:
				loss = self._train_step(imgs,labels)

				if self.training_steps_done %10 ==0:
					self.log_loss(self.training_steps_done,loss,log_path)
				self.training_steps_done += 1
				batches_done+=1
				if batches_done % 500 == 0:
					print('Epoch: ',epoch,' batches done: ',batches_done,'out of ',self.train_data_size//self.batch_size)

			print('Epoch:',epoch ,' done')

			print(f'Calculating accuracy')
			accuracy = utils.accuracy_on_dataset(self.classifier, self.test_dataset)
			print(f'Accuracy on test set: {accuracy}')
			if log_test_dataset_accuracy_after_epoch:
				self.log_accuracy(epoch, accuracy, log_path)

			if accuracy > bestAccuracy and save_best_classified_model is True:
				shutil.rmtree(classifier_save_path, ignore_errors=True)
				self.classifier.saveWeights(classifier_save_path+save_name)
				bestAccuracy = accuracy
		#Save model po poslednom epochu, ak nechce nalepsi
		if not save_best_classified_model:
			shutil.rmtree(classifier_save_path, ignore_errors=True)
			self.classifier.saveWeights(classifier_save_path + save_name)


		print('Training finished')
		print(f'Best accuracy on test set is {bestAccuracy}')


	def log_accuracy(self,epoch,acc,log_path):
		if epoch == 0:
			utils.append_to_file(f"{log_path}trainlog_accuracy.txt", "epoch:accuracy\n")

		utils.append_to_file(f"{log_path}trainlog_accuracy.txt", f"{epoch}:{acc}\n")


	def log_loss(self,iteration,loss,log_path):
		if iteration == 0:
			utils.append_to_file(f"{log_path}trainlog_loss.txt", "iteration:loss\n")

		utils.append_to_file(f"{log_path}trainlog_loss.txt", f"{iteration}:{loss}\n")











