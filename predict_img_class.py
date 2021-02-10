"""make predictions with the classifier model for yelp photos dataset"""

from src import classifier
import argparse
import skimage.io
import numpy as np
import keras
from skimage.transform import resize
import matplotlib.pyplot as plt



DIM=128
CATEGORIES = {0: "food", 1: "inside", 2: "outside", 3: "drink", 4: "menu"}

def preprocess_img(image):
	"""preprocess image."""
	img = skimage.io.imread(image, plugin='pil')
	resized = resize(img, (DIM, DIM), anti_aliasing=True, preserve_range=True)
	scaled = resized / 127.5 - 1
	return scaled


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Predict label for a given input image')
	parser.add_argument('--image', required=True, help='Path to input image')	

	args = parser.parse_args()

	# load trained model
	model = keras.models.load_model("files/trained_model.h5")

	# read input image
	image = preprocess_img(args.image)
	image = np.expand_dims(image, 0)

	# make predictions
	prediction = model.predict(image)
	category = np.argmax(prediction, -1)[0]

	# name
	name = CATEGORIES[category]

	print('The given input image {} belongs to category: {}'.format(args.image, name))

	plt.imshow(image[0])
	plt.show()







