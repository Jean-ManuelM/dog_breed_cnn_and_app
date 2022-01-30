# Dog Breed Classification Project declined in a app

## Table of Contents
- Introduction & Motivation
- Libraries used
- Model and function creations
- Main difficuly
- Web Application
- Instructions to run the Web application
- Ressources

## Introduction & Motivation 
This project uses Convolutional Neural Networks (CNNs)! In this project,we build a pipeline to process real-world, user-supplied images.
Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

To complete this approach, you can play with this algorithm in a web application.

## Libraries used
import json

import plotly

import pandas as pd

import nltk

from sklearn.datasets import load_files      

from keras.utils import np_utils

import numpy as np

from glob import glob


import cv2             

import matplotlib.pyplot as plt 

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image         

from tqdm import tqdm

from keras.applications.resnet50 import preprocess_input, decode_predictions


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint  

from extract_bottleneck_features import *

from flask import Flask

from flask import render_template, request, jsonify

You need to donwload in store in feature the the file following

## Model creation

You can find all the steps related to the creation and of the differents functions and model necessary to our task in the associated notebook dog_app.ipynb.

The main steps are :
- Import the Dataset
- Create a Human Detector Humans 
- Create a dog Detector Humans 
- Create a CNN to classify Breeds, from scratch --> 2% of accurency
- Use a CNN to Classify Dog Breeds, with Transfer Learning --> 65 % of accurency
- Create a CNN to Classify Dog Breeds, , with Transfer Learning with specialise model --> 80% of accurency
- Sum-up the all in a function 

## Main difficuly
### Problem Introduction
To find the correct dog breed.

For human and others types, it is not important because it more for fun than a science.
The function to detect Human face and detect a dog are already suit by Udacity.

### Strategy to solve the problem
 To solve the problem, u used in order these strategie :
 
 - Use Udacity learning :
 - Experience : With my experience with CNN for image processing in my work, I know that 
 - Online documentation : I search advice online and took it (See Ressources section)
 - Test and retry : Due to the facts 
   - There is few parameters
   - The amount of time to a set of parameter (with a correct epochs) take only a minute 
   - The results was at the rendezvous quickly.
 
 
### Metrics
accurency
### EDA
### Modelling
Benchmark Analysis is the base model against which your model could be compared. For example, if you are building a binary classification model using random forest, then the benchmark model could be logistic regression, which is a simple model. In your case, for dog breed your benchmark model could be some simple neural network model or any of the neural network model on which you can compare your current model and prove that your current model is better than your benchmark model.

A decision tree could also work for this type of categorisation problem.

### Hyperparameter tuning
Hyper parameters are the parameters which you will using for your neural nework model. You could find detailed explanation of hyper paramters in below link,

https://towardsdatascience.com/neural-networks-parameters-hyperparameters-and-optimization-strategies-3f0842fac0a5
- I don't use Grid CV
### Results - Conclusion/Reflection
The results became very quicky 
None of the images I looked give me a strong wrong answer.
I was able to detect than a wolf is not a dog ! I was very impress.

But I think there are some idea to improve the model :

### Improvements
- Add a metadata : the sex of the dog.
- Improved the dataset, especcially for close breeds :
- Use InceptionV3
- Use GridCV



## Web Application
To use this model, you can find a web app running with Flask.

### Instructions to run the Web application

1.You can clone the current repository.

2.You need to download the Resnet50 model at this adress https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz and store this file as "DogResnet50Data.npz" in the folder "bottleneck_features".

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Ressources
https://knowledge.udacity.com/

https://keras.io/api/

https://www.simplilearn.com/tutorials/deep-learning-tutorial/guide-to-building-powerful-keras-image-classification-models
