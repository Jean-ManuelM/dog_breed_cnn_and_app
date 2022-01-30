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
To find the correct dog breed from a picture, we need to create a model really performant.
But how to to that :

Note : <i> For human and others types classification in dog breed, it is not important because it more for fun than a science.
The functions to detect Human face and detect a dog are already suit by Udacity. </i>

### Strategy to solve the problem and model seletion
 To solve the problem, u used in order these strategie :
 
 - Use Udacity learning : udacity encourage us to use CNN to answer to this method.
   - They also provided us an exemaple of a
   - They also provided us an exemaple of pre-trained model and the gain in term of accurency.
   - Therefore, we continue on this way.
   -
 - Experience : With my experience with CNN for image processing in my work, I know that " branch model with direct, semi direct and 3 layers models give good results and are simple. This is often a good combinaison to get to get the first steps of good performance.
 - Online documentation : I search advice online and took it on how choose as parameter for this type of classsifcation. (See Ressources section)
 - Test and retry : Due to the facts 
   - There is few parameters
   - The amount of time to a set of parameter (with a correct epochs) take only a minute 
   - The results was at the rendezvous quickly.

A decision tree could also work for this type of categorisation problem.

We can use a optimiser like GridSearchCV to find the best configuration.

### Actual Architecture
- I start by initialisaing the model downloaded (resnet_50) as Sequencial model.
- I define an input_shape with the right format for the selected model.
- I define the associated Dense paremeter (number of node) with the number of breed in our dataset = 133.
- By experience of the previous step, I test to add a dropout to limited the overfitting and it was an improvement for the value 0,05. 0,1 and 0,2 was not.
- Optimiser : I test to change the optimizer in 'adam' but'rmsprop' seem better.
- Choice of pre-trained CNN : I tried to download the InceptionV3 but it was too big in term of MB. Therefore I go for the Resnet_50

### Data 
Our Datasets have been provided by Udacity :  
We split our data into too group by random selection : train and a test set.

- There are 133 total dog categories.
- There are 8351 total dog images.
- There are 6680 training dog images.
- There are 835 validation dog images.
- There are 836 test dog images.

### Metrics
The accurency is our main metric.
This is the percentage of correct prediction made by the model compared to the test set.


### Results - Conclusion/Reflection
The results became very quicky 
None of the images I looked give me a strong wrong answer.
I was able to detect than a wolf is not a dog ! I was very impress.

But I think there are some idea to improve the model :

### Improvements
- Add a metadata in the data: the sex of the dog.
Because the sex change sometime the appereace of a dog, he may confuse the model sometime. As example for the Beauceron breed, the male are 10-15 kg giger than the female and have they ear cut because they are much taller.

![Beauceron male](https://www.dog-breeds-expert.com/images/beauceron-1.jpg)

![Beauceron female](https://i68.servimg.com/u/f68/18/16/32/88/iuka0110.jpg)

- Improved the dataset, get more picture, especcially for close breeds : My model give me as result that this beauceron is a doberman... and i can't blame it !
- 
![Beauceron male](https://www.dog-breeds-expert.com/images/beauceron-1.jpg)

![Doberman](https://www.photos-nature-passion.fr/images/photo-de-chien-doberman_3.jpg)


- Use InceptionV3
It seem to be a greatest pre-trained model for this type of classification.

- Use GridsearchCV (or equivalent)
To find the best parameters.



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
