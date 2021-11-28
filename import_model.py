
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalMaxPooling2D
from image_processing import plot_images , load_image, img_path 
import numpy as np
import cv2
import swifter
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def get_embedding(DATASET_PATH, model, img_name, img_width, img_height):
    # Reshape
    img = image.load_img(img_path(DATASET_PATH,img_name), target_size=(img_width, img_height))
    # img to Array
    x   = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)

def map_embedding(DATASET_PATH, model, img_width, img_height,df ):
    # Parallel apply
    map_embeddings = df['image'].swifter.apply(lambda img: get_embedding(DATASET_PATH, model, img, img_width, img_height))
    df_embs        = map_embeddings.apply(pd.Series)
    return df_embs

def find_cosine_similarity(df_embs):
    # Calcule DIstance Matriz
    cosine_sim = 1-pairwise_distances(df_embs, metric='cosine')
    return cosine_sim

def get_recommender(cosine_sim, indices, idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim
    