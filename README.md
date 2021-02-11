# Yelp

This is a simple POC project exploring yelp dataset [](https://www.yelp.com/dataset).

Exploratory data analysis and processing including recommendation algorithms are done in jupyter-notebooks in the `notebooks` directory.

Clone the repository locally and install required libraries using:

```
pip install -r requirements.txt
```

There are two pickled Machine Learning model files that are large, so to test this repo, please download them from [here](https://seafile.cloud.uni-hannover.de/d/26b531317733439b82b4/).

There are two files names:
- `user_based_recommender_nn`
- `item_based_recommender_nn`

that should be downloaded and put under `files` directory

If you don't want to clone the repo locally, you could download the `Dockerfile` and the `requirements.txt` in this repository and build a docker image. The dockerfile automatically installs necessary libraries and clones this repository. 

Having downloaded the `Dockerfile` and the `requirements.txt` in a local empty directory (along with the downloaded model files), build a docker image using:

```
docker build -t <YOUR-CHOICE-OF-NAME>:<YOUR-CHOICE-OF-TAG> .
```

Then run:

```
docker run -it <YOUR-CHOICE-OF-NAME>:<YOUR-CHOICE-OF-TAG> bash
```

In your local machine or the docker container you could test different programs explained below:


There are three different machine learning tasks applied:

## Image Classification

MobileNetV2 model in Keras is fine-tuned to classify images in the yelp photos dataset into 5 categories: food, inside, outside, drink and menu.

training is done using the command:

```
python train_img_classifier.py
```

The trained weights and pickled model are included in the `files` directory.

It could be tested using the command:

```
python predict_img_class.py --image=/path/to/image
```

It will print out the category predicted for the given image. Additionally, it will open an image window showing the given image.

## Rating Prediction from Review Text

Users have given a review for a business and rated them with the 1-5 rating scale.

Machine learning techniques such as TFIDF and classification algorithms such s Random Forest in Scikit-learn are used to predict a star rating a user might give based on the review text.

The model is trained using the command:

```
python train_rating_predictor.py
```

The trained models are pickled in the `files` directory. They could be tested using the command:

```
python predict_review_rating.py
```

It will ask the user to input some text review and the model will predict the star rating for it.


## Business (e.g., restaurant) recommender
Based on the user reviews for business, another machine learning task is recommending businesses to users. It could be done using similarities between users who might have liked (rated similarly) similar businesses or based on similar restaurants the user might have rated (liked).

Two different approaches are experimented: Memory Based (KNeearest Neighbour) and Model Based (SVD or Singular Value Decomposition). For these tasks, the Scikit-surprise library is used. Models are tested on small chunk of the huge dataset as I have a simple machine. Jupyter-notebooks for analysis and training the models are given in the `notebooks` directory. Trained models are pickled and saved in the `files` directory.

The models could be tested using the command:

```
python recommend_business.py
```
It will ask the user to input a user ID (that exists in the dataset, small subset of it is included in `files/subsets.csv`) and the business ID to score, it will predict possible rating score and give top recommendations to the user.


Please let me know if there are any issues/mistakes you notice, Thanks.

