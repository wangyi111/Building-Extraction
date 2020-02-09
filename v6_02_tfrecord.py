
# coding: utf-8

# In[1]:

##数据集统一处理，将图像数据和标签放在一起生成二进制文件，能更好的利用内存
import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np


# In[2]:



#数据集路径
DATASET_DIR_TRAIN_IMG="D:/jupyter_pycode/buildings/33333/clipping_data/Input Images/"  
DATASET_DIR_TRAIN_LABEL="D:/jupyter_pycode/buildings/33333/clipping_data/Target Maps/"



#txt路径
TXT_DIR_TRAIN="D:/jupyter_pycode/buildings/33333/clipping_data/train.txt"




#tfrecord文件存放路径
TFRECORD_DIR="D:/jupyter_pycode/buildings/33333/"

#判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train']:
        output_filename=os.path.join(dataset_dir,split_name+'.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
        return True
    
#获取图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames=[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path=os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    return photo_filenames

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values=[values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_number,image_name,image_data,label_data):
    #abstract base class for protocol messages
    return tf.train.Example(features=tf.train.Features(feature={
        'order':int64_feature(image_number), #int
        'name':bytes_feature(image_name),   #str
        'image':bytes_feature(image_data),  #'rgb'
        'label':bytes_feature(label_data),  #'1'

                
            }))


##读取图片名txt文件
def read_examples_list(path):

    """
    Args:
      path: absolute path to examples list file.
  
    Returns:
      list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]



#把数据转换为tfrecord格式
def _convert_dataset(split_name,image_dir,label_dir,examples):
    assert split_name in ['train']
    
    with tf.Session() as sess:
        #定义tfrecord文件的路径+名字
        output_filename=os.path.join(TFRECORD_DIR,split_name+'.tfrecords')
        
      
        
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,example in enumerate(examples):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(examples)))
                    sys.stdout.flush()
                    
                    image_path = os.path.join(image_dir, example + '.png')
                    label_path = os.path.join(label_dir, example + '.png')
                    
                    image_name=bytes(example,encoding='utf8')
                    
                    image_number=i
           
                    #读取图片
                    image_data=Image.open(image_path)
                    #转为矩阵
                    image_data=np.array(image_data)
                    #将图片转为bytes
                    image_data=image_data.tobytes()
                    
                    
                    #读取标签
                    label_data=Image.open(label_path)                  
                    #转为矩阵
                    label_data=np.array(label_data)
                    #转为bytes
                    label_data=label_data.tobytes()
                
       
                    #生成protocol数据类型
                    tf_example=image_to_tfexample(image_number,image_name,image_data,label_data)
                    tfrecord_writer.write(tf_example.SerializeToString())
                    
                except IOError as e:
                    print('Could not read:',filename)
                    print('Error:',e)
                    prin('Skip it\n')
            sys.stdout.write('\n')
            sys.stdout.flush()



#判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print('tfrecord文件已存在')
else:

    
    #读取txt
    train_examples=read_examples_list(TXT_DIR_TRAIN)
    #test_examples=read_examples_list(TXT_DIR_TEST)
    
    #数据转换
    _convert_dataset('train',DATASET_DIR_TRAIN_IMG,DATASET_DIR_TRAIN_LABEL,train_examples)
    #_convert_dataset('test',DATASET_DIR_TEST_IMG,DATASET_DIR_TEST_LABEL,test_examples)
    print('生成tfrecord文件')


# In[ ]:



