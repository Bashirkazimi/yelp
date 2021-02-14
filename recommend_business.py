"""restaurant recommender"""

# from src import classifier
# import argparse
# import skimage.io
# import numpy as np
# import keras
# from skimage.transform import resize
# import matplotlib.pyplot as plt

import os
from surprise import dump

from collections import defaultdict



if __name__ == "__main__":

	# make predictions using SVD model
	_, svd_model = dump.load('files/svd_restaurant_recommender')

	uid = input("Please enter a user id (text, contained in the /files/subset.csv file)\n")
	bid = input("Please enter a business id you want to get recommendations for (text, contained in the /files/subset.csv file)\n")

	score = svd_model.predict(uid, bid).est

	print('Possible rating by the user to the given business predicted by SVD model: {}'.format(score))

	# make predictions using user-based nn recommender
	_, userBasedKNN = dump.load('files/user_based_recommender_nn')

	score = userBasedKNN.predict(uid, bid).est

	print('Possible rating by the user to the given business predicted by user based recommender model: {}'.format(score))


	# make predictions using item-based nn recommender
	_, itemBasedKNN = dump.load('files/item_based_recommender_nn')

	score = itemBasedKNN.predict(uid, bid).est
	print('Possible rating by the user to the given business predicted by item based recommender model: {}'.format(score))












