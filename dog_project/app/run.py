import json
import plotly
import pandas as pd
import nltk
#nltk.download()

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

import cv2                
import matplotlib.pyplot as plt                        
#%matplotlib inline  

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  

#from extract_bottleneck_features import *
def extract_Resnet50(tensor):
	from keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar


import joblib
from sqlalchemy import create_engine



#functions

def face_detector(img_path):
    """  returns "True" if human face is detected in image stored at img_path """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


""" define ResNet50 model """
ResNet50_model = ResNet50(weights='imagenet')


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    """ returns prediction vector for image located at img_path"""
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    """ returns "True" if a dog is detected in the image stored at img_path"""
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def load_model():
    model_Resnet50_1 = Sequential()

    model_Resnet50_1.add(GlobalAveragePooling2D(input_shape=train_Resnet50_1.shape[1:]))
    
    model_Resnet50_1.add(Dense(133, activation= 'softmax'))

    model.add(Dropout(0.05))

    model_Resnet50_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model_Resnet50_1.load_weights('saved_models/weights.best.Resnet50_1.hdf5')
    return model_Resnet50_1
    
    
def Resnet_50_1_predict_breed(img_path):
    """Function that takes a path to an image as input and returns the dog breed that is predicted by the model."""
    
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path)) 
    
    # obtain predicted vector
    predicted_vector = model_Resnet50_1.predict(bottleneck_feature)
    
    # return dog breed that is predicted by the model in a ggod shape
    breed = dog_names[np.argmax(predicted_vector)]
    breed = breed.rsplit(".")[1]
    return breed

def what_a_dog(img_path):
    """Input is a  image. 
    first determines whether the image contains a human, dog, or neither. Then,

    if a dog is detected in the image, return the predicted breed.
    if a human is detected in the image, return the resembling dog breed.
    if neither is detected in the image, provide output that indicates an error."""
    
    if face_detector(img_path) == True:
        Message1 = "This is a human. His closer dog breed is : "
    if dog_detector(img_path) == True:
        Message1 = "This is dog. His breed must be : "
    if face_detector(img_path)== False and dog_detector(img_path) == False:
        Message1 = "This is not dog or not human. It must be a mistake, only dogs and humans exist... The closer dog breed : "
    
    Message1 = Message1 + Resnet_50_1_predict_breed(img_path)
    
    return Message1


UPLOAD_FOLDER = './images/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')
    
    
       
    
  

"""# Web page that handles user query and displays model results"""

@app.route('/go', methods = ['GET', 'POST'])
def go():
   
    
    if request.method == 'POST':
        image_nom = secure_filename(f.filename)
        print(image_nom)
        
        image = request.files['file']
        
        
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_nom))
        
        
        answer = what_a_dog(os.path.join(app.config['UPLOAD_FOLDER'], image_nom))

     
    return render_template(
        'go.html',
        answer=answer
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
#host='127.0.0.1', port=5000

if __name__ == '__main__':
    main()