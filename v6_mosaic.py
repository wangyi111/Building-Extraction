
# coding: utf-8

# In[1]:

##影像拼接

from PIL import Image
import numpy as np
import os
import sys


DATASET_DIR="D:/jupyter_pycode/buildings/33333/prediction_clipping/"
OUTPUT_DIR="D:/jupyter_pycode/buildings/33333/prediction_total/"


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
    
    target=Image.new('RGB',(1952,2944))
    for i,filename in enumerate(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
        
        #获取文件名标签
        labels=filename.split('/')[-1][0:28]
        
        labels_1=labels[0:22]
        labels_2=labels[23:28]
        
        
        
        img=Image.open(filename) 
        img_size=img.size
        x0=0
        y0=0
        width=16
        height=16
        stride=16
        
        for x_num in range(122):
            for y_num in range(184):
                x=x0+x_num*stride
                y=y0+y_num*stride
                w=width
                h=height
                region_name=x_num*184+y_num  #竖着存            
                
                if int(labels_2)==region_name:
                    target.paste(img,(x,y,x+w,y+h))
                    
        outfile=os.path.join(output_dir,labels_1+'.png')            
        target.save(outfile)
        
      
                

        
    sys.stdout.write('\n')
    sys.stdout.flush()
  



    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR)
_img_mosaic(photo_filenames1,DATASET_DIR,OUTPUT_DIR)    

print('complete.')

    
    
    
    

