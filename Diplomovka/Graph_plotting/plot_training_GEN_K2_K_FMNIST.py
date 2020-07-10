
import numpy as np
import matplotlib.pyplot as plt
path_accuracy = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/trainlog_class_accuracy.txt')
path_gen_loss = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/trainlog_gen_loss.txt')
path_discr_loss = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/trainlog_discr_loss.txt')
path_eucl = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/EUCL.txt')
path_manh = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/MANH.txt')
path_mse = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/MSE.txt')
path_psnr = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/PSNR.txt')
path_ssim = ('../GANs/experiments/trainings_best_classified/GENERATOR_K2_withClass_K_FMNIST/log/SSIM.txt')

def decode_log_file(path):
	with open(path, "r") as f1:
		lines = f1.readlines()
		lines = lines[1:]
		x, y = [list(d) for d in zip(*[[i for i in c.split(':')] for c in lines])]
		x = list(map(int, x))
		y = list(map(float, y))
		return np.array(x), np.array(y)

def decode_log_file_metric(path):
	with open(path, "r") as f1:
		lines = f1.readlines()
		lines = lines[1:]
		x, y,y_av = [list(d) for d in zip(*[[i for i in c.split(':')] for c in lines])]
		x = list(map(int, x))
		y_av = list(map(float, y_av))
		return np.array(x), np.array(y_av)


def make_acc(x_acc,y_acc,ax):
	ax.plot(x_acc, y_acc, marker='o')
	max_acc_index = np.argmax(y_acc)
	ax.plot(x_acc[max_acc_index], y_acc[max_acc_index], "gs", markersize=5)
	ax.set_yticks(np.concatenate((np.arange(0, 1.1, 0.10), np.arange(0.9, 1, 0.025))))
	ax.set_title('Vývoj presnosti klasifikácie zväčšených obrázkov generátorom GENERATOR_K2 na K-FMNIST-2\n(Klasická chybová funkcia geenrátora (G_loss)) ')
	ax.set_xlabel("Počet vykonaných epochov\n1 epoch = 1875 tréningových krokov (batchov)")
	ax.set_ylabel("Presnosť na testovacej množine")
	ax.grid(axis='y', linestyle='-')
	for i, j in zip(x_acc, y_acc):
		ax.annotate('{:.4f}'.format(j), xy=(i, j), xytext=(5, 5), textcoords='offset points')

def plot_metric(x,y,ax,title,highlightLowest=True):
	ax.plot(x, y, marker='o')
	highlight_index = np.argmin(y)
	if not highlightLowest:
		highlight_index = np.argmax(y)

	ax.plot(x[highlight_index], y[highlight_index], "gs", markersize=5)
	# ax.set_yticks(np.concatenate((np.arange(0, 1.1, 0.10), np.arange(0.9, 1, 0.025))))
	ax.set_title(title)
	ax.set_xlabel("Počet vykonaných epochov")
	ax.set_ylabel("Hodnota metriky")
	ax.grid(axis='y', linestyle='-')
	ax.annotate('{:.4f}'.format(y[highlight_index]), xy=(x[highlight_index], y[highlight_index]), xytext=(3, 3), textcoords='offset points')

def plot_loss(x,y,ax,title,titleY):
	ax.set_xlabel("Počet vykonaných tréningových krokov")
	ax.set_ylabel(titleY)
	ax.set_title(title)
	ax.plot(x, y)



fig, ax = plt.subplots(1,1)
fig  ,(ax1,ax2) = plt.subplots(1,2)
fig, axs = plt.subplots(2,2)

x_acc,y_acc = decode_log_file(path_accuracy)
x_gen_loss,y_gen_loss = decode_log_file(path_gen_loss)
x_discr_loss,y_discr_loss= decode_log_file(path_discr_loss)
x_eucl,y_eucl = decode_log_file_metric(path_eucl)
x_mse,y_mse= decode_log_file_metric(path_mse)
x_psnr,y_psnr= decode_log_file_metric(path_psnr)
x_ssim,y_ssim= decode_log_file_metric(path_ssim)


make_acc(x_acc,y_acc,ax)

plot_loss(x_gen_loss,y_gen_loss,ax1,'Vývoj hodnoty chybovej funkcie generátora','Hodnota chybovej funkcie')
plot_loss(x_discr_loss,y_discr_loss,ax2,'Vývoj hodnoty chybovej funkcie diskriminátora','Hodnota chybovej funkcie')

plot_metric(x_eucl,y_eucl,axs[0,0],'Vývoj euklidovskej vzdialenosti')
plot_metric(x_mse,y_mse,axs[0,1],'Vývoj strednej kvadratickej chyby (MSE)')
plot_metric(x_ssim,y_ssim,axs[1,0],'Vývoj SSIM indexu',highlightLowest=False,)
plot_metric(x_psnr,y_psnr,axs[1,1],'Vývoj PSNR metriky',highlightLowest=False,)

plt.show()




