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


# taken from: https://surprise.readthedocs.io/en/stable/FAQ.html
def get_top_n(predictions, n=10):
	"""Return the top-N recommendation for each user from a set of predictions.

	Args:
		predictions(list of Prediction objects): The list of predictions, as
			returned by the test method of an algorithm.
		n(int): The number of recommendation to output for each user. Default
			is 10.

	Returns:
	A dict where keys are user (raw) ids and values are lists of tuples:
		[(raw item id, rating estimation), ...] of size n.
	"""

	# First map the predictions to each user.
	top_n = defaultdict(list)
	for uid, iid, true_r, est, _ in predictions:
		top_n[uid].append((iid, est))

	# Then sort the predictions for each user and retrieve the k highest ones.
	for uid, user_ratings in top_n.items():
		user_ratings.sort(key=lambda x: x[1], reverse=True)
		top_n[uid] = user_ratings[:n]

	return top_n



if __name__ == "__main__":

	# make predictions using SVD model
	_, svd_model = dump.load('files/svd_restaurant_recommender')

	uid = input("Please enter a user id (text, contained in the /files/subset.csv file)\n")
	bid = input("Please enter a business id you want to get recommendations for (text, contained in the /files/subset.csv file)\n")

	dummy_rate = 0

	input_ = [(uid, bid, dummy_rate)]

	predictions = svd_model.test(input_)

	top_n = get_top_n(predictions, n=1)

	
	for uid, user_ratings in top_n.items():
		for i, (iid, score) in enumerate(user_ratings):
			#print('\t{} {}'.format(iid, score))
			print('Possible rating the user to the given business predicted by SVD model: {}'.format(score))

	# make predictions using user-based nn recommender
	_, userBasedKNN = dump.load('files/user_based_recommender_nn')

	predictions = userBasedKNN.test(input_)
	top_n = get_top_n(predictions, n=5)

	print('top recommendations to the user by KNN user based recommender')
	for uid, user_ratings in top_n.items():
		for i, (iid, score) in enumerate(user_ratings):
			print('\t{} {}'.format(iid, score))


	# make predictions using item-based nn recommender
	_, itemBasedKNN = dump.load('files/item_based_recommender_nn')

	predictions = itemBasedKNN.test(input_)
	top_n = get_top_n(predictions, n=5)

	print('top recommendations to the user by KNN item based recommender')
	for uid, user_ratings in top_n.items():
		for i, (iid, score) in enumerate(user_ratings):
			print('\t{} {}'.format(iid, score))











