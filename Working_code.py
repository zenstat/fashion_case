import os
import time
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from import_python import  set_up_working_directory, data_basic_details
from import_spark import  spark_spin_up_session,spark_read_images_from_path, spark_return_image_attribute
from image_processing import plot_images, load_image,  img_path,  resnet_model, get_embedding
import numpy as np
import cv2
from tensorflow.keras.models import load_model
 

#spark_read_images_from_path(spark, "e:/dev/Kaggle/fashion/sample/1164.jpg")
DATASET_PATH = "e:/dev/Kaggle/fashion/"
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=8000, error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
#load 
emb1= np.loadtxt('embed.txt', dtype=int)
df_embs1= np.loadtxt('df_embs.txt', dtype=int)
model1 = load_model('my_model.h5')

from sklearn.metrics.pairwise import pairwise_distances
# Calcule DIstance Matriz
cosine_sim = 1-pairwise_distances(df_embs1, metric='cosine')
cosine_sim[:4, :4]

indices = pd.Series(range(len(df)), index=df.index)
indices

# Function that get movie recommendations based on the cosine similarity score of movie genres
def get_recommender(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim

output = get_recommender(2993, df, top_n = 6)