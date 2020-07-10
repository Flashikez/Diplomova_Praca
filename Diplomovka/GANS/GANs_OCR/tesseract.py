import numpy as np
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
import re
import tensorflow as tf


# Mapovanie cisel na pismena
num_to_char_map = {"0": "O",
		   "1": "I",
		   "2": "Z",
		   "4": "A",
		   "5": "S",
		   "8": "B",}
# Mapovanie pismen na cisla
char_to_num_map = dict([value,key] for key,value in num_to_char_map.items())

# Vráti rozpoznaný reťazec
def ocr_text(image,whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",psm = "AUTO"):

	return tess.image_to_string(image, config=f"-c tessedit_char_whitelist={whitelist}")





def unnormalize(image):
	image = image * 127.5 + 127.5
	return tf.round(image)

def cast(data,dtype=tf.dtypes.uint8):
	return tf.cast(data,dtype)

# Na každý znak stringu aplikuje mapovanie
def apply_mapping(string,mapping):
	ret = ""
	for c in string:
		if c in mapping:
			ret = ret + mapping[c]
		else:
			ret = ret + c
	return ret




# Formátuje string na formát ECV
def format_as_ecv(string):
	sub =  re.sub(r"(..)(...)(..)(.*$)", r"\1 \2 \3", string)
	sub = sub.upper()
	if len(sub) == 9:
		sub = apply_mapping(sub[0:2],num_to_char_map) +" "+apply_mapping(sub[3:6],char_to_num_map)+ " " +apply_mapping(sub[7:9],num_to_char_map)

	return sub

# Threshold funkcia na dáta obrázka
def threshhold(img_data,mode = cv2.THRESH_BINARY):
	retval,image = cv2.threshold(img_data,127,255,mode)
	return image

def numpy_data_bw(image):
	# image = cast(image,dtype=tf.dtypes.uint8)
	return image.numpy()[:,:,0]

