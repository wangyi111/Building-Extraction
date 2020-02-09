
# coding: utf-8

# In[1]:

##影像拼接

from PIL import Image,ImageDraw
import numpy as np
import os
import sys


DATASET_DIR="D:/jupyter_pycode/buildings/33333/png_data/Target Maps/"
OUTPUT_DIR="D:/jupyter_pycode/buildings/33333/Target Maps/"


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
    
    #target=Image.new('RGB',(1952,2944))
    for i,filename in enumerate(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
        
        #获取文件名标签
        labels=filename.split('/')[-1][0:22]
        
        labels_1=labels[0:22]
        #labels_2=labels[23:28]
        
        
        
        img=Image.open(filename) 
        img_size=img.size
        width=img_size[0]
        height=img_size[1]
        print(width,height)
        
        target=Image.new('RGB', (width, height), (0,0,0))
        draw = ImageDraw.Draw(target)
        
        lab=np.asarray(img)
        print(lab.shape)
        #lab=Image.fromarray(img)
        
        for x in range(height):
            for y in range(width):
            
                
                if lab[x,y]==1:
                    draw.point((y, x), fill=(255,0,0))
                    
        outfile=os.path.join(output_dir,labels_1+'.png')            
        target.save(outfile)
        
      
                

        
    sys.stdout.write('\n')
    sys.stdout.flush()
  



    
photo_filenames1=_get_filenames_and_classes(DATASET_DIR)
_img_mosaic(photo_filenames1,DATASET_DIR,OUTPUT_DIR)    

print('complete.')

    
    
    
    

