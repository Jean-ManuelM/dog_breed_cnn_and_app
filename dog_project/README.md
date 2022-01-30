# Dog Breed Classification Project declined in a app

## Table of Contents
Introduction & Motivation
Libraries used
Model and function creations
Web Application
Instructions to run the Web application
Ressources

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
