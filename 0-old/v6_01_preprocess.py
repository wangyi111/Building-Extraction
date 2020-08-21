
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np
from scipy import misc  ##misc：以图像形式保存矩阵  scipy：科学计算库
from libtiff import TIFF  ##处理tiff格式




# In[2]:

DATASET_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/33333/initial_data/Target Maps/"
OUTPUT_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/33333/png_data/Target Maps/"

def tiff_to_image_array(tiff_image_name, labels, out_folder, out_type):   
            
    tif = TIFF.open(tiff_image_name, mode = "r")  
    idx = 0  
    for im in list(tif.iter_images()):  
        #  
        im_name = out_folder + str(labels) + out_type  
        misc.imsave(im_name, im)  
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
    
    
    
photo_filenames2=_get_filenames_and_classes(DATASET_DIR_TRAIN_LABEL)
_tif_to_png(photo_filenames2,DATASET_DIR_TRAIN_LABEL,OUTPUT_DIR_TRAIN_LABEL)











# In[3]:

##影像裁剪

from PIL import Image
import numpy as np
import os
import sys


DATASET_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/33333/png_data/Input Images/"
OUTPUT_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/33333/clipping_data/Input Images/"


#获取所有图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames=[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path=os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    return photo_filenames

#创建文件夹（可选）
def mkdir(path):
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False



def _img_clipping(filenames,dataset_dir,output_dir):
    for i,filename in enumerate(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
        
        #获取文件名标签
        labels=filename.split('/')[-1][0:22]
        
        img=Image.open(filename)
        img_size=img.size
        x0=0
        y0=0
        width=64
        height=64
        stride=16
        img_clipped=[]
        
        x_iter=(img_size[0]-width+stride)//stride
        y_iter=(img_size[1]-height+stride)//stride
        #2
        #path=os.path.join(output_dir,labels)
        #mkdir(path)
        
        for x_num in range(x_iter):
            for y_num in range(y_iter):
                x=x0+x_num*stride
                y=y0+y_num*stride
                w=width
                h=height
                region=img.crop((x,y,x+w,y+h))
                
                region_name=str(x_num*y_iter+y_num)  #竖着存
                
                #1：全部存到一个文件夹下
                #outfile=os.path.join(output_dir,labels + '_' + region_name + '.jpg')
                
                if len(region_name)==1:
                    outfile=os.path.join(output_dir,labels + '_' + '0000' + region_name + '.png')
                if len(region_name)==2:
                    outfile=os.path.join(output_dir,labels + '_' + '000' + region_name + '.png')
                if len(region_name)==3:
                    outfile=os.path.join(output_dir,labels + '_' + '00' + region_name + '.png')
                    
                if len(region_name)==4:
                    outfile=os.path.join(output_dir,labels + '_' + '0'+ region_name + '.png')
                if len(region_name)==5:
                    outfile=os.path.join(output_dir,labels + '_' + region_name + '.png')
                
                
                
                
                
                
                
                #2：存到137个子文件夹下
                #outfile=os.path.join(path,labels + '_' + region_name + '.jpg')
                
                region.save(outfile)
                
                
                
                
                
                #region=np.array(region.convert('L'))
                #img_clipped.append(region)
                

        
    sys.stdout.write('\n')
    sys.stdout.flush()
  



    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR_TRAIN_IMG)
_img_clipping(photo_filenames1,DATASET_DIR_TRAIN_IMG,OUTPUT_DIR_TRAIN_IMG)    


    
    
    
    


# In[6]:

##标签裁剪

from PIL import Image
import numpy as np
import os
import sys


DATASET_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/33333/png_data/Target Maps/"
OUTPUT_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/33333/clipping_data/Target Maps/"


#获取所有图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames=[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path=os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    return photo_filenames

#创建文件夹（可选）
def mkdir(path):
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False



def _img_clipping(filenames,dataset_dir,output_dir):
    for i,filename in enumerate(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
        
        #获取文件名标签
        labels=filename.split('/')[-1][0:22]
        
        img=Image.open(filename)
        img_size=img.size
        x0=24
        y0=24
        width=16
        height=16
        stride=16
        img_clipped=[]
        
        x_iter=(img_size[0]-64+stride)//stride
        y_iter=(img_size[1]-64+stride)//stride
        #2
        #path=os.path.join(output_dir,labels)
        #mkdir(path)
        
        for x_num in range(x_iter):
            for y_num in range(y_iter):
                x=x0+x_num*stride
                y=y0+y_num*stride
                w=width
                h=height
                region=img.crop((x,y,x+w,y+h))
                
                region_name=str(x_num*y_iter+y_num)  #竖着存
                
                #1：全部存到一个文件夹下
                #outfile=os.path.join(output_dir,labels + '_' + region_name + '.jpg')
                
                if len(region_name)==1:
                    outfile=os.path.join(output_dir,labels + '_' + '0000' + region_name + '.png')
                if len(region_name)==2:
                    outfile=os.path.join(output_dir,labels + '_' + '000' + region_name + '.png')
                if len(region_name)==3:
                    outfile=os.path.join(output_dir,labels + '_' + '00' + region_name + '.png')
                    
                if len(region_name)==4:
                    outfile=os.path.join(output_dir,labels + '_' +'0'+region_name + '.png')
                
                if len(region_name)==5:
                    outfile=os.path.join(output_dir,labels + '_' +region_name + '.png')
                
                
                
                
                
                
                
                #2：存到137个子文件夹下
                #outfile=os.path.join(path,labels + '_' + region_name + '.jpg')
                
                region.save(outfile)
                
                
                
                
                
                #region=np.array(region.convert('L'))
                #img_clipped.append(region)
                

        
    sys.stdout.write('\n')
    sys.stdout.flush()
  



    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR_TRAIN_LABEL)
_img_clipping(photo_filenames1,DATASET_DIR_TRAIN_LABEL,OUTPUT_DIR_TRAIN_LABEL)    


    
    
    
    


# In[7]:

##生成训练文件名
#图片名存入txt文件
import os
import sys
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png':  
                # L.append(os.path.join(root, file))  
                file_name = file[0:-4]  #去掉.txt
                L.append(file_name)  
    return L  
 
label_folder = 'D:/jupyter_pycode/buildings/33333/clipping_data/Input Images/'
trainval_file = 'D:/jupyter_pycode/buildings/33333/clipping_data/train.txt'
 
txt_name = file_name(label_folder)
k=0 
with open(trainval_file, 'w') as f:
    for i in txt_name:
        f.write('{}\n'.format(i))
        k=k+1
        sys.stdout.write('\r>> Converting image %d / %d' % (k,len(txt_name)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
f.close()

print('complete.')


# In[ ]:



