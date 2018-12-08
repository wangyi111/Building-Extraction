
# coding: utf-8

# In[ ]:

##影像拼接

from PIL import Image
import numpy as np
import os
import sys


DATASET_DIR="D:/jupyter_pycode/buildings/00000trial/clipping_data/Train Set/Input Images/"
OUTPUT_DIR="D:/jupyter_pycode/buildings/00000trial/prediction_total/"


#获取所有图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames=[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path=os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    return photo_filenames

#拼接
def _img_mosaic(filenames,dataset_dir,output_dir):
    
    target=Image.new('RGB',(1440,1440))
    for i,filename in enumerate(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
        
        #获取文件名标签
        labels=filename.split('/')[-1][0:16]
        
        labels_1=labels[0:11]
        labels_2=labels[12:16]
        
        
        
        img=Image.open(filename) 
        img_size=img.size
        x0=0
        y0=0
        width=64
        height=64
        stride=16
        
        for x_num in range(90):
            for y_num in range(90):
                x=x0+x_num*stride
                y=y0+y_num*stride
                w=width
                h=height
                region_name=x_num*90+y_num  #竖着存            
                
                if int(labels_2)==region_name:
                    target.paste(img,(x,y,x+w,y+h))
                    
        outfile=os.path.join(output_dir,labels_1+'.png')            
        target.save(outfile)
        
      
                

        
    sys.stdout.write('\n')
    sys.stdout.flush()
  



    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR)
_img_mosaic(photo_filenames1,DATASET_DIR,OUTPUT_DIR)    



    
    
    
    

