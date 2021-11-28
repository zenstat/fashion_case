
import pandas as pd
import numpy as np
import cv2
import os as os
from tensorflow.keras.models import load_model
from import_python import  set_up_working_directory, data_basic_details
from import_spark import  spark_spin_up_session,spark_read_images_from_path, spark_return_image_attribute
from image_processing import plot_images, load_image,  img_path,  resnet_model, get_embedding
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
DATASET_PATH = "e:/dev/Kaggle/fashion/"
    
def get_images(idx_ref, top_n = 5):
    # Recommendations

    os.chdir(DATASET_PATH)
    emb1= np.loadtxt("embed.txt", dtype=int)   
    model1 = load_model("my_model.h5")
    df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=8000, error_bad_lines=False)
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = df.reset_index(drop=True)
    idx_rec, idx_sim = get_recommender(idx_ref, df, top_n = 6)
    plt.imshow(cv2.cvtColor(load_image(DATASET_PATH,df.iloc[idx_ref].image), cv2.COLOR_BGR2RGB))
    figures = {'im'+str(i): load_image(DATASET_PATH,row.image) for i, row in df.loc[idx_rec].iterrows()}
    plot_images(figures, 2, 3)

def get_recommender(idx, df ,top_n = 5):

    # Calcule DIstance Matriz
    df_embs1= np.loadtxt("df_embs.txt", dtype=int)
    
    cosine_sim = 1-pairwise_distances(df_embs1, metric='cosine')
    indices = pd.Series(range(len(df)), index=df.index)
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
get_images( 2993)