FROM python:3.6-stretch
MAINTAINER Tina Bu <tina.hongbu@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# check our python environment
RUN python3 --version
RUN pip3 --version


RUN git clone https://github.com/Bashirkazimi/yelp.git

WORKDIR /yelp

#COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY user_based_recommender_nn /files/user_based_recommender_nn
COPY item_based_recommender_nn /files/item_based_recommender_nn


