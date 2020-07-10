
import numpy as np
import matplotlib.pyplot as plt
path_GEN_TOP= ('../GANs/GANs_OCR/experiments/GENERATOR_K2_TextSet_extended/tesseract_logs/CHARS_STATS/GENERATOR_TOP.txt')
path_LANCZOS3 = ('../GANs/GANs_OCR/experiments/GENERATOR_K2_TextSet_extended/tesseract_logs/CHARS_STATS/LANCZOS3.txt')

def decode_alphabet_file(path):
	with open(path, "r") as f1:
		lines = f1.readlines()
		lines = lines[1:]
		alphabet, total,correct,wrong = [list(d) for d in zip(*[[i for i in c.split(':')] for c in lines])]
		total = list(map(int, total))
		correct = list(map(int, correct))
		wrong = list(map(int, wrong))
		alphabet[26] = "medzera"
		return np.array(alphabet),np.array(total), np.array(correct),np.array(wrong)




def make_acc(x_acc,y_acc,ax):
	ax.plot(x_acc, y_acc, marker='o')
	idx = (-y_acc).argsort()[:3]
	# max_acc_index = np.argmax(y_acc)
	ax.plot(x_acc[idx[0]], y_acc[idx[0]], "gs", markersize=5)
	ax.plot(x_acc[idx[1]], y_acc[idx[1]], "gs", markersize=5)
	ax.plot(x_acc[idx[2]], y_acc[idx[2]], "gs", markersize=5)

	# ax.set_yticks(np.concatenate((np.arange(0, 1.1, 0.10), np.arange(0.9, 1, 0.025))))
	ax.set_xticks(np.arange(0,21,1))
	ax.set_title('Vývoj SSIM indexu zväčšených obrázkov generátorom GENERATOR_K2 na rozšírenom datasete  Text_Set\n(Klasická chybová funkcia geenrátora (G_loss)) ')
	ax.set_xlabel("Počet vykonaných epochov\n1 epoch = 7500 tréningových krokov (batchov)")
	ax.set_ylabel("Presnosť na testovacej množine")
	ax.grid(axis='y', linestyle='-')
	for x,(i, j) in enumerate(zip(x_acc, y_acc)):
		if x in idx:
			if x == idx[1]:
				ax.annotate('{:.6f}'.format(j), xy=(i, j), xytext=(5, -15), textcoords='offset points')
			else:
				ax.annotate('{:.6f}'.format(j), xy=(i, j), xytext=(5, 5), textcoords='offset points')






plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
# width = 0.35



gen_alp,gen_total,gen_correct,gen_wrong = decode_alphabet_file(path_GEN_TOP)
_,lancz_total,lancz_correct,lancz_wrong = decode_alphabet_file(path_LANCZOS3)
gen_alp_AZ = gen_alp[:26]
gen_total_AZ = gen_total[:26]
gen_wrong_AZ = gen_wrong[:26]

lancz_total_AZ = lancz_total[:26]
lancz_wrong_AZ = lancz_wrong[:26]

arrangeX = np.arange(len(gen_alp_AZ))
rects1 = ax.bar(arrangeX-0.3,gen_total_AZ,width=0.3,label="Celkový počet")
rects2 = ax.bar(arrangeX,gen_wrong_AZ,width=0.3,label="Počet nesprávne rozpoznaných znakov zväčšených generátorom")
rects3 = ax.bar(arrangeX+0.3,lancz_wrong_AZ,width=0.3,label="Počet nesprávne rozpoznaných znakov zväčšených LANCZOS3 ")


ax.set_ylabel('Počet')
ax.set_xlabel('Znak')
ax.set_title('Počty nesprávne rozpoznaných znakov zväčšeného rozšíreného testovacieho datasetu Text_Set')
ax.set_xticks(arrangeX)
ax.set_xticklabels(gen_alp_AZ)
ax.legend()


gen_alp_AZ = gen_alp[26:]
gen_total_AZ = gen_total[26:]
gen_wrong_AZ = gen_wrong[26:]

lancz_total_AZ = lancz_total[26:]
lancz_wrong_AZ = lancz_wrong[26:]

arrangeX = np.arange(len(gen_alp_AZ))
rects1 = ax2.bar(arrangeX-0.3,gen_total_AZ,width=0.3,label="Celkový počet")
rects2 = ax2.bar(arrangeX,gen_wrong_AZ,width=0.3,label="Počet nesprávne rozpoznaných znakov zväčšených generátorom")
rects3 = ax2.bar(arrangeX+0.3,lancz_wrong_AZ,width=0.3,label="Počet nesprávne rozpoznaných znakov zväčšených LANCZOS3 ")


ax2.set_ylabel('Počet')
ax2.set_xlabel('Znak')
ax2.set_title('Počty nesprávne rozpoznaných znakov zväčšeného rozšíreného testovacieho datasetu Text_Set')
ax2.set_xticks(arrangeX)
ax2.set_xticklabels(gen_alp_AZ)
ax2.legend()


plt.show()




