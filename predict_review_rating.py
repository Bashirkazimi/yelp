"""make predictions with the classifier model for yelp photos dataset"""

import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import joblib
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer




if __name__ == "__main__":


	# load the model from disk
	loaded_model = joblib.load('files/RandomForest1.pkl')
	
	rawtext = input("Please enter a review:\n")

	stemmer = WordNetLemmatizer()
	text = re.sub(r'\W', ' ', str(rawtext))
	text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
	text = re.sub(r'\s+', ' ', text, flags=re.I)
	text = re.sub(r'^b\s+', '', text)
	text = text.lower()
	text = text.split()
	text = [stemmer.lemmatize(word) for word in text]
	text = ' '.join(text)

	rating = loaded_model.predict([text])[0]


	print('The predicted rating for the review: {} is: {} stars!'.format(rawtext, rating+1))









