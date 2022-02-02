# Dog Breed Classification Project declined in a app

## Table of Contents
- Introduction & Motivation
- Libraries used
- Project Writeup
- Web Application
- Instructions to run the Web application
- Ressources

## Introduction & Motivation 
This project uses Convolutional Neural Networks (CNNs)! In this project,we build a pipeline to process real-world, user-supplied images.
Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

To complete this approach, you can play with this algorithm in a web application.

## Libraries used
- json
- plotly
- pandas 
- nltk
- sklearn.datasets import load_files      
- keras.utils import np_utils
- numpy as np
- glob import glob
- cv2             
- matplotlib.pyplot as plt 
- keras.applications.resnet50 import ResNet50
- keras.preprocessing import image         
- tqdm import tqdm
- keras.applications.resnet50 import preprocess_input, decode_predictions
- keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
- keras.layers import Dropout, Flatten, Dense
- keras.models import Sequential
- keras.callbacks import ModelCheckpoint  
- extract_bottleneck_features import *
- flask import Flask
- flask import render_template, request, jsonify

## Project Writeup
You wil find the project Writeup in the root repository.
It contains all the explanations needed to understand the projects.



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
