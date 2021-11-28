 
import pyspark
import os as os
from pyspark.sql import SparkSession

def spark_spin_up_session(session_name):
    spark = SparkSession.builder.appName(session_name).getOrCreate()
    return spark 

def spark_read_csv(sparkobj, file_path):
    spark = sparkobj
    readobj= spark.read.option('header','true').csv(file_path)
    return readobj

def spark_read_images_from_path(sparkobj, file_path):
    spark = sparkobj
    image_df = spark.read.format("image").load(file_path, inferschema=True)
    return image_df

def spark_return_image_attribute(image_obj):
    image_obj.select("image.origin", "image.width", "image.height","image.nChannels", "image.mode").show(truncate=False)







 

