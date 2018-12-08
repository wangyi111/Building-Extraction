
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


# In[ ]:

##01-1 批量转换图片格式

DATASET_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/00001trial/initial_data/Train Set/Input Images/"
OUTPUT_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/00001trial/png_data/Train Set/Input Images/"



DATASET_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/00001trial/initial_data/Train Set/Target Maps/"
OUTPUT_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/00001trial/png_data/Train Set/Target Maps/"


DATASET_DIR_TEST_IMG="D:/jupyter_pycode/buildings/00001trial/initial_data/Test Set/Input Images/"
OUTPUT_DIR_TEST_IMG="D:/jupyter_pycode/buildings/00001trial/png_data/Test Set/Input Images/"



DATASET_DIR_TEST_LABEL="D:/jupyter_pycode/buildings/00001trial/initial_data/Test Set/Target Maps/"
OUTPUT_DIR_TEST_LABEL="D:/jupyter_pycode/buildings/00001trial/png_data/Test Set/Target Maps/"


# DATASET_DIR=[]
# DATASET_DIR.append(DATASET_DIR_TRAIN_IMG,DATASET_DIR_TRAIN_LABEL,DATASET_DIR_TEST_IMG,DATASET_DIR_TEST_LABEL)
# OUTPUT_DIR=[]
# OUTPUT_DIR.append(OUTPUT_DIR_TRAIN_IMG,OUTPUT_DIR_TRAIN_LABEL,OUTPUT_DIR_TEST_IMG,OUTPUT_DIR_TEST_LABEL)

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
        labels=filename.split('/')[-1][0:11]
        
        tiff_to_image_array(filename,labels,output_dir,'.png')
    sys.stdout.write('\n')
    sys.stdout.flush()
        
        
        
        
        
photo_filenames1=_get_filenames_and_classes(DATASET_DIR_TRAIN_IMG)
_tif_to_png(photo_filenames1,DATASET_DIR_TRAIN_IMG,OUTPUT_DIR_TRAIN_IMG)
        
photo_filenames2=_get_filenames_and_classes(DATASET_DIR_TRAIN_LABEL)
_tif_to_png(photo_filenames2,DATASET_DIR_TRAIN_LABEL,OUTPUT_DIR_TRAIN_LABEL)

photo_filenames3=_get_filenames_and_classes(DATASET_DIR_TEST_IMG)
_tif_to_png(photo_filenames3,DATASET_DIR_TEST_IMG,OUTPUT_DIR_TEST_IMG)

photo_filenames4=_get_filenames_and_classes(DATASET_DIR_TEST_LABEL)
_tif_to_png(photo_filenames4,DATASET_DIR_TEST_LABEL,OUTPUT_DIR_TEST_LABEL)


# In[ ]:

##01-2-1 影像裁剪

from PIL import Image
import numpy as np
import os
import sys


DATASET_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/00001trial/png_data/Train Set/Input Images/"
OUTPUT_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/00001trial/clipping_data/Train Set/Input Images/"

DATASET_DIR_TEST_IMG="D:/jupyter_pycode/buildings/00001trial/png_data/Test Set/Input Images/"
OUTPUT_DIR_TEST_IMG="D:/jupyter_pycode/buildings/00001trial/clipping_data/Test Set/Input Images/"

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
        labels=filename.split('/')[-1][0:11]
        
        img=Image.open(filename)
        img_size=img.size
        x0=0
        y0=0
        width=64
        height=64
        stride=16
        img_clipped=[]
        

        for x_num in range(90):
            for y_num in range(90):
                x=x0+x_num*stride
                y=y0+y_num*stride
                w=width
                h=height
                region=img.crop((x,y,x+w,y+h))
                
                region_name=str(x_num*90+y_num)  #竖着存
                                
                if len(region_name)==1:
                    outfile=os.path.join(output_dir,labels + '_' + '000' + region_name + '.png')
                if len(region_name)==2:
                    outfile=os.path.join(output_dir,labels + '_' + '00' + region_name + '.png')
                if len(region_name)==3:
                    outfile=os.path.join(output_dir,labels + '_' + '0' + region_name + '.png')
                if len(region_name)==4:
                    outfile=os.path.join(output_dir,labels + '_' + region_name + '.png')
                

                region.save(outfile)
                

    sys.stdout.write('\n')
    sys.stdout.flush()
  



    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR_TRAIN_IMG)
_img_clipping(photo_filenames1,DATASET_DIR_TRAIN_IMG,OUTPUT_DIR_TRAIN_IMG)    


photo_filenames2=_get_filenames_and_classes(DATASET_DIR_TEST_IMG)
_img_clipping(photo_filenames2,DATASET_DIR_TEST_IMG,OUTPUT_DIR_TEST_IMG)
    
    
    
    


# In[ ]:

##01-2-2 标签裁剪
from PIL import Image,ImageDraw
import numpy as np
import os
import sys



DATASET_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/00001trial/png_data/Train Set/Target Maps/"
OUTPUT_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/00001trial/clipping_data/Train Set/Target Maps/"

DATASET_DIR_TEST_LABEL="D:/jupyter_pycode/buildings/00001trial/png_data/Test Set/Target Maps/"
OUTPUT_DIR_TEST_LABEL="D:/jupyter_pycode/buildings/00001trial/clipping_data/Test Set/Target Maps/"

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
        labels=filename.split('/')[-1][0:11]
        
        img0=Image.open(filename)
        img0_size=img0.size
        
        
        ###【v2】二值化
        pix = img0.load()
        width = img0_size[0]
        height = img0_size[1]
        img = Image.new('L', (width, height), (0))
        draw = ImageDraw.Draw(img)
        for x in range(width):
            for y in range(height):
                r,g,b=pix[x,y]
                #rr=r/255
                if r==255:
                    draw.point((x, y), fill=1)
        
        
        x0=24
        y0=24
        width=16
        height=16
        stride=16
        img_clipped=[]
        
        #裁剪
        for x_num in range(90):
            for y_num in range(90):
                x=x0+x_num*stride
                y=y0+y_num*stride
                w=width
                h=height
                region=img.crop((x,y,x+w,y+h))
                
                region_name=str(x_num*90+y_num)  #竖着存
                
                ###【v2】文件名改进
                if len(region_name)==1:
                    outfile=os.path.join(output_dir,labels + '_' + '000' + region_name + '.png')
                if len(region_name)==2:
                    outfile=os.path.join(output_dir,labels + '_' + '00' + region_name + '.png')
                if len(region_name)==3:
                    outfile=os.path.join(output_dir,labels + '_' + '0' + region_name + '.png')
                if len(region_name)==4:
                    outfile=os.path.join(output_dir,labels + '_' + region_name + '.png')
                
                region.save(outfile)
                
                

        
    sys.stdout.write('\n')
    sys.stdout.flush()
    

    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR_TRAIN_LABEL)
_img_clipping(photo_filenames1,DATASET_DIR_TRAIN_LABEL,OUTPUT_DIR_TRAIN_LABEL)

photo_filenames2=_get_filenames_and_classes(DATASET_DIR_TEST_LABEL)
_img_clipping(photo_filenames2,DATASET_DIR_TEST_LABEL,OUTPUT_DIR_TEST_LABEL)

    




# In[ ]:

##01-3-1 生成训练文件名
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
 
label_folder = 'D:/jupyter_pycode/buildings/00001trial/clipping_data/Train Set/Input Images/'
trainval_file = 'D:/jupyter_pycode/buildings/00001trial/clipping_data/Train Set/train.txt'
 
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

##01-3-2 生成测试文件名
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
 
label_folder = 'D:/jupyter_pycode/buildings/00001trial/clipping_data/Test Set/Input Images/'
trainval_file = 'D:/jupyter_pycode/buildings/00001trial/clipping_data/Test Set/test.txt'
 
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

