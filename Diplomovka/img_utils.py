import tensorflow as tf


# Zmení rozlíšenie obrázkov, používané na zmenšenie rozlíšenia
def downsample(imgs,size):
	return tf.image.resize(imgs,size)
# Normalizácia čiernobielých obrázkov do intervalu <-1;1>
def normalize(imgs):
	return (imgs - 127.5) / 127.5
# Zmení rozlíšenie obrázkov, používané na zväčšenie rozlíšenia metódou method
def upsample(imgs,size,method):
	return tf.image.resize(imgs,size,method=method)
# Opak normalizácie, teda zmení interval hodnôt späť na <0;255> ak sa hodnoty nachádzali v intervale <-1;1>
def unnormalize(imgs):
	return (imgs * 127.5)+127.5
