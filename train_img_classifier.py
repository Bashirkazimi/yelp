"""train classifier model for yelp photos dataset"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

from src import classifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import skimage.io
import keras
from skimage.transform import resize
import tensorflow as tf


BATCH_SIZE = 64 # batch size
DIM = 128 # input dimension


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


if __name__ == "__main__":

	keras.backend.clear_session()
	# create new session
	config = tf.ConfigProto()
	# config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	keras.backend.set_session(tf.InteractiveSession(config=config))

	directory = '/home/bashir/Desktop/yelp/data/photos'
	df = pd.read_json('/home/bashir/Desktop/yelp/data/photos.json', lines=True)


	# data is imbalanced, calcualte class weights
	num_examples = df.value_counts('label').tolist()
	total = np.sum(num_examples)
	weights = [(1./ne)*total/2.0 for ne in num_examples]

	class_weights = {}
	for i in range(len(weights)):
		class_weights[i] = weights[i]


	# split data for training, testing and validation
	df_train, df_valid = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
	df_train, df_test = train_test_split(df_train, test_size=0.1, stratify=df_train['label'], random_state=42)

	# 
	# df_train, df_valid = train_test_split(df, test_size=0.7, stratify=df['label'], random_state=42)
	# df_train, df_test = train_test_split(df_train, test_size=0.1, stratify=df_train['label'], random_state=42)
	# df_train, df_valid = train_test_split(df_train, test_size=df_test.shape[0], stratify=df_train['label'], random_state=42)

	print(df_train.shape, df_test.shape, df_valid.shape)


	# prepare data
	ds_train = YelpData(directory, df_train, BATCH_SIZE, True, dim=DIM)
	ds_valid = YelpData(directory, df_valid, BATCH_SIZE, False, dim=DIM)
	ds_test = YelpData(directory, df_test, BATCH_SIZE, False, dim=DIM)



	# create and compile the model
	model = classifier.create_model(input_shape=(DIM, DIM, 3), num_classes=5)
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'],
	)

	# checkpoints to save only weights
	checkpoint = tf.keras.callbacks.ModelCheckpoint('files/best_weights.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

	# train the model
	hist = model.fit_generator(ds_train, 
		epochs=5, 
		validation_data=ds_valid, 
		callbacks=[checkpoint], 
		# use_multiprocessing=True,
		# workers=4,
		class_weight=class_weights)

	# save the trained model
	model.load_weights('files/best_weights.h5')
	model.save('files/trained_model.h5')

	# save training history to csv
	df = pd.DataFrame(hist.history)
	df.to_csv('files/train_history.csv', index=False)

	# evaluate on test data
	res = model.evaluate_generator(ds_test)
	print('Loss on test data: {}\nAccuracy on test data: {}'.format(res[0], res[1]))




