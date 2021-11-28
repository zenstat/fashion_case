 
import os
import time
from PIL import Image
import pandas as pd


def set_up_working_directory(str_dir_path):
    os.chdir(str_dir_path)
    print("Working direcotry changed to " + os.getcwd())

def data_basic_details(image_folder_path): 
    
    df = pd.DataFrame(columns = ['FileName', 'width', 'height'])
    print(df)
    start_time = time.time()
    #iterate through every file
    for r, d, f in os.walk(image_folder_path):
        for file in f:
            if file.endswith(".jpg"):
                width, height = Image.open(os.path.join(r, file)).size
                df = df.append({'FileName' :  file , 'width' :width  , 'height' :height  },ignore_index = True)
    #total time taken 
    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))
