"""make predictions with the classifier model for yelp photos dataset"""

from src import classifier
import argparse
import skimage.io
import numpy as np
import keras
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class YelpData(keras.utils.Sequence):
	"""Yelp Dataset Class"""
	def __init__(self, input_dir, df, chunk_size, shuffle, dim=128):
		self.input_dir = input_dir
		self.photo_ids = df.photo_id.tolist()
		self.labels = df.label.tolist()
		self.id_to_label = dict(zip(self.photo_ids, self.labels))
		self.categories =  {'food': 0, 'inside': 1, 'outside': 2, 'drink': 3, 'menu': 4}
		self.chunk_size = chunk_size
		self.shuffle = shuffle
		self.dim = dim
	
	def __len__(self):
		return int(np.ceil(len(self.labels) / self.chunk_size))
	
	def __getitem__(self, index):
		photo_ids = self.photo_ids[index*self.chunk_size:(index+1)*self.chunk_size]
		return self.generate_batch(photo_ids)

	def on_epoch_end(self):
		if self.shuffle == True:
			np.random.shuffle(self.photo_ids)

	def generate_batch(self, photo_ids):
		"""generate one batch """
		batch_x = np.empty((len(photo_ids), self.dim, self.dim, 3))
		batch_y = np.empty(len(photo_ids),)

		for i, photo_id in enumerate(photo_ids):
			img = skimage.io.imread(os.path.join(self.input_dir, photo_id+'.jpg'), plugin='pil')
			label = self.id_to_label[photo_id]
			resized = resize(img, (self.dim, self.dim), anti_aliasing=True, preserve_range=True)
			scaled = resized / 127.5 - 1 
			batch_x[i] = scaled
			batch_y[i] = self.categories[label]

		return batch_x, batch_y


DIM=128
CATEGORIES = {0: "food", 1: "inside", 2: "outside", 3: "drink", 4: "menu"}




if __name__ == "__main__":

	keras.backend.clear_session()
	# create new session
	config = tf.ConfigProto()
	# config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	keras.backend.set_session(tf.InteractiveSession(config=config))

	directory = '/home/bashir/Desktop/yelp/data/photos'
	df = pd.read_json('/home/bashir/Desktop/yelp/data/photos.json', lines=True)


	# split data for training, testing and validation
	df_train, df_valid = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
	df_train, df_test = train_test_split(df_train, test_size=0.1, stratify=df_train['label'], random_state=42)

	df_test = df_test.iloc[:200]
	ds_test = YelpData(directory, df_test, 200, False, dim=DIM)
	for x, y in ds_test:
		break


	# load trained model
	model = keras.models.load_model("files/trained_model.h5")


	# make predictions
	prediction = model.predict(x, batch_size=200)
	category = np.argmax(prediction, -1)

	print('TEST RESULTS')
	print(classification_report(y, category))


