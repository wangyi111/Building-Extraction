
# coding: utf-8

# In[1]:

import tensorflow as tf
from nets import resnet_v2 
import os
from PIL import Image,ImageDraw
import numpy as np
import time
import sys


# In[2]:

start=time.clock()
predict_dir="D:/jupyter_pycode/buildings/11111/v555_resnet/predict_result/"
label_dir="D:/jupyter_pycode/buildings/11111/v555_resnet/targetmap/"
output_dir="D:/jupyter_pycode/buildings/11111/v555_resnet/accuracy.txt"
def _get_filenames_and_classes(dataset_dir):
    photo_filenames=[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path=os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    return photo_filenames

def predict_accuracy(filenames,predict_dir,label_dir,output_dir):
    acc_txt=[]
    acc_txt.append('total_accuracy, miss_err, error_err:')
    total_nums=[]
    miss_nums=[]
    error_nums=[]
    label_nums=[]
    predict_nums=[]
    for i,filename in enumerate(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
        sys.stdout.flush()
         #获取文件名标签
        example=filename.split('/')[-1][0:11]
        
        predict_path = os.path.join(predict_dir, example + '.png')
        label_path = os.path.join(label_dir, example + '.png')
        
        predict_data=Image.open(predict_path)
        label_data=Image.open(label_path)
        #predict_size=predict_data.size
        predict_size=predict_data.size
        width2=predict_size[0]
        height2=predict_size[1]
        pix2=predict_data.load()
        pix1=label_data.load()
        total_num=0
        miss_num=0
        error_num=0
        label_num=0
        predict_num=0
        for x in range(width2):
            for y in range(height2):
                r1,g1,b1=pix1[x+24,y+24] #true label
                r2,g2,b2=pix2[x,y]  #prediction
                
                if r1==r2:
                    total_num=total_num+1
                if r1==255:
                    label_num=label_num+1
                if r2==255:
                    predict_num=predict_num+1
                if r1==255 and r2==0:
                    miss_num=miss_num+1
                if r1==0 and r2==255:
                    error_num=error_num+1
                    
        total_accuracy=total_num/(width2*height2)
        miss_err=miss_num/label_num
        error_err=error_num/predict_num
        
        total_nums.append(total_num)
        miss_nums.append(miss_num)
        error_nums.append(error_num)
        label_nums.append(label_num)
        predict_nums.append(predict_num)
        
        
        
        #acc=[]
        acc_txt.append(str(total_accuracy)+' '+str(miss_err)+' '+str(error_err))
        
        #acc_txt.append(acc)
    
    all_accuracy=sum(total_nums)/(1440*1440)
    all_miss_err=sum(miss_nums)/sum(label_nums)
    all_error_err=sum(error_nums)/sum(predict_nums)
    
    acc_txt.append('************************************************************')
    acc_txt.append('all_accuracy, all_miss_err, all_error_err:')
    acc_txt.append(str(all_accuracy)+' '+str(all_miss_err)+' '+str(all_error_err))
    
    sys.stdout.write('\n')
    sys.stdout.flush()        
    
    
    with open(output_dir, 'w') as f:
        for i in acc_txt:
            f.write('{}\n'.format(i))
    f.close()
    

    
    
photo_filenames1=_get_filenames_and_classes(predict_dir)
#photo_filenames2=_get_filenames_and_classes(label_dir)
        
predict_accuracy(photo_filenames1,predict_dir,label_dir,output_dir)    

print('complete.')

end=time.clock()
print(end-start,' s')
                
        


# In[ ]:



