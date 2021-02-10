import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
nltk.download('wordnet')
nltk.download('stopwords')
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
# this part is copied from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from scipy.stats import randint, truncnorm, uniform
from sklearn.model_selection import RandomizedSearchCV



if __name__ == "__main__":
	# read reviews chunk by chunk as they don't fit into memory
	df = pd.read_json("/data/yelp_academic_dataset_review.json", orient="records", lines=True, chunksize=100000)

	# create review-rating pairs and clean review texts.
	reviews = []
	stars = []
	stemmer = WordNetLemmatizer()
	chunk_counter = 0
	for chunk in df:
		chunk_stars = chunk['stars'].tolist()
		chunk_texts = chunk['text'].tolist()
		for cs, ct in zip(chunk_stars, chunk_texts):
			text = re.sub(r'\W', ' ', str(ct))
			text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
			text = re.sub(r'\s+', ' ', text, flags=re.I)
			text = re.sub(r'^b\s+', '', text)
			text = text.lower()
			text = text.split()
			text = [stemmer.lemmatize(word) for word in text]
			text = ' '.join(text)
			
			# add preprocessed review
			reviews.append(text)
			
			# add star rating
			stars.append(cs-1)
		chunk_counter += 1
		if chunk_counter %100 == 0:
			print('{} chunks done!'.format(chunk_counter))
		break

	# put the review-rating pair to dataframe
	df = pd.DataFrame({"reviews": reviews, "stars": stars})

	# Create a pipeline for feature extraction and classification.
	mypipeline = Pipeline(
		[
			('countVectorizer', CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))),
			('tfidfTransformer', TfidfTransformer()),
			('classifier', RandomForestClassifier(random_state = 42))
		]
	)

	# split data into train/test (80/20 percent) 
	reviews, reviews_test, stars, stars_test = train_test_split(reviews, stars, stratify=stars, test_size=0.2,random_state=42)

	# train classifier
	mypipeline.fit(reviews, stars)

	# make predictions on test data
	prediction = mypipeline.predict(reviews_test)

	# evaluate
	print(classification_report(stars_test, prediction))

	# save classifier in pickle file
	joblib.dump(mypipeline, '/files/RandomForest1.pkl', compress = 1)


	# test some other classifiers
	names = ["Nearest Neighbors",
			 "Decision Tree", "Random Forest", "AdaBoost",
			 ]

	classifiers = [
		KNeighborsClassifier(10),
		DecisionTreeClassifier(max_depth=50),
		RandomForestClassifier(random_state = 42),
		AdaBoostClassifier()
	]

	# run, print evaluation results and save models to file!
	for name, classifier in zip(names, classifiers):
		mypipeline = Pipeline(
			[
				('countVectorizer', CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))),
				('tfidfTransformer', TfidfTransformer()),
				('classifier', classifier)
			]
		)
		mypipeline.fit(reviews, stars)
		prediction = mypipeline.predict(reviews_test)
		print('\n classifier: {}'.format(name))
		print(classification_report(stars_test, prediction))
		joblib.dump(mypipeline, '/files/{}.pkl'.format(name), compress = 1)



	# finally, do some cross validation and use randomizedsearch to find best parameters	
	model_params = {
		'classifier__n_estimators': randint(4,200),
		'classifier__max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
		'classifier__min_samples_split': uniform(0.01, 0.199)
	}

	mypipeline = Pipeline(
		[
			('countVectorizer', CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))),
			('tfidfTransformer', TfidfTransformer()),
			('classifier', RandomForestClassifier(random_state = 42))
		]
	)

	clf = RandomizedSearchCV(mypipeline, model_params, n_iter=100, cv=5, random_state=1)
	clf.fit(reviews, stars)

	joblib.dump(clf.best_estimator_, '/files/bestRFclassifier.pkl', compress = 1)
	