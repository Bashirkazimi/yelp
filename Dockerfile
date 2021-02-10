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

# set the working directory for containers

# Installing python dependencies
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN cd /home


# Copy all the files from the project’s root to the working directory
# COPY src/ /src/
# RUN ls -la /src/*

RUN git clone https://github.com/Bashirkazimi/BashirLearning.git

# Running Python Application
# CMD ["python3", "/src/main.py"]
