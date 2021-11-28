import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalMaxPooling2D

 

def plot_images(figures, nrows = 1, ncols=1,figsize=(8, 8)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    
def img_path(DATASET_PATH , img):
    return DATASET_PATH+"images/"+img

def load_image(DATASET_PATH, img):
    return cv2.imread(img_path(DATASET_PATH, img))

def resnet_model(DATASET_PATH, df):
    # Input Shape
    img_width, img_height, _ = load_image(DATASET_PATH,df.iloc[0].image).shape
    
    # Pre-Trained Model
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape = (img_width, img_height, 3))
    base_model.trainable = False
    
    # Add Layer Embedding
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

    
def get_embedding(DATASET_PATH,  model, img_name):
    # Reshape
    img = image.load_img(  img_path(DATASET_PATH, img_name), target_size=(img_width, img_height))
    # img to Array
    x   = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)
    
    
    
    
    
    
    
    