
import numpy as np
import matplotlib.pyplot as plt
path_loss = ('../Classifiers/K_F_MNIST/training_log/FMNIST/trainlog_loss.txt')
path_accuracy = ('../Classifiers/K_F_MNIST/training_log/FMNIST/trainlog_accuracy.txt')

def decode_log_file(path):
	with open(path, "r") as f1:
		lines = f1.readlines()
		lines = lines[1:]
		x, y = [list(d) for d in zip(*[[i for i in c.split(':')] for c in lines])]
		x = list(map(int, x))
		y = list(map(float, y))
		return np.array(x), np.array(y)



x_loss,y_loss = decode_log_file(path_loss)
x_acc,y_acc = decode_log_file(path_accuracy)
print(x_acc,y_acc)


fig, (ax1,ax2) = plt.subplots(1,2)
ax1.set_xlabel("Počet vykonaných tréningových krokov")
ax1.set_ylabel("Hodnota  (Kategorická krížová entropia)")
ax1.plot(x_loss, y_loss)
ax1.set_title('Vývoj hodnoty chybovej funkcie počas tréningu K-FMNIST-2')
ax2.plot(x_acc,y_acc,marker='o')
max_acc_index = np.argmax(y_acc)
ax2.plot(x_acc[max_acc_index],y_acc[max_acc_index],"gs",markersize=5)
ax2.set_yticks(np.concatenate((np.arange(0, 1.1, 0.10),np.arange(0.9,1,0.025))))
ax2.set_title('Vývoj presnosti klasifikátora K-FMNIST-2 na testovacej množine počas tréningu')
ax2.set_xlabel("Počet vykonaných epochov\n1 epoch = 1875 tréningových krokov (batchov)")
ax2.set_ylabel("Presnosť na testovacej množine")
ax2.grid(axis='y',linestyle='-')
for i,j in zip(x_acc,y_acc):
	ax2.annotate('{:.4f}'.format(j),xy=(i,j),xytext=(5,5), textcoords='offset points')

plt.show()




