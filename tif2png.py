import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np
from scipy import misc  ##misc：以图像形式保存矩阵  scipy：科学计算库
from libtiff import TIFF  ##处理tiff格式


DATASET_DIR_TRAIN_IMAGE="D:/Git_projects/Bachelor-thesis/vaihingen/ISPRS_semantic_labeling_Vaihingen/top/"
OUTPUT_DIR_TRAIN_IMAGE="D:/Git_projects/Bachelor-thesis/new/train_image/"

DATASET_DIR_TRAIN_LABEL= 'D:/Git_projects/Bachelor-thesis/vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/'
OUTPUT_DIR_TRAIN_LABEL='D:/Git_projects/Bachelor-thesis/new/train_label/'
def tiff_to_image_array(tiff_image_name, labels, out_folder, out_type):   
            
    tif = TIFF.open(tiff_image_name, mode = "r")  
    idx = 0  
    for im in list(tif.iter_images()):  
        #  
        im_name = out_folder + str(labels) + out_type  
        #img = im.read_image()
        png = Image.fromarray(im)
        if not os.path.isfile(im_name):
            png.save(im_name)
        #misc.imsave(im_name, im)  
        #print(im_name, 'successfully saved!!!')  
        idx = idx + 1  
    return 


#获取所有图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames=[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path=os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    return photo_filenames

def _tif_to_png(filenames,dataset_dir,output_dir):
    for i,filename in enumerate(filenames):
        
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
        #获取label
        labels=filename.split('/')[-1][0:22]
        
        tiff_to_image_array(filename,labels,output_dir,'.png')
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    
    
#photo_filenames2=_get_filenames_and_classes(DATASET_DIR_TRAIN_IMAGE)
#_tif_to_png(photo_filenames2,DATASET_DIR_TRAIN_IMAGE,OUTPUT_DIR_TRAIN_IMAGE)

photo_filenames1=_get_filenames_and_classes(DATASET_DIR_TRAIN_LABEL)
_tif_to_png(photo_filenames1,DATASET_DIR_TRAIN_LABEL,OUTPUT_DIR_TRAIN_LABEL)