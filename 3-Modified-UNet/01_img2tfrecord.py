import os
import tensorflow as tf
from PIL import Image
import glob
from skimage.transform import resize
from skimage import io
import numpy as np
import time

"""
注意，以下程序中img为0-255，label为0,1（建筑物）。生成tfrecords没有归一化
"""

def img2TFRecord(img_dir,label_dir,tfrecord_file,num_examples_txt_file):
    """
    将图像和对应的标签制作成TFRecord文件，之后使用tf.data.TFRecordDataset读取TFRecord文件制作成dataset
    :img_dir: 图像的文件夹
    :label_dir: 标签的文件夹
    :tfrecord_file: tfrecord文件，如 "train.tfrecords"
    :num_examples_txt_file: 存放样本数目的txt文件
    """
    # 要生成的tfrecord文件
    print("prepare to generate %s..."%(tfrecord_file))
    t0 = time.time()
    writer = tf.io.TFRecordWriter(tfrecord_file)
    img_filenames = glob.glob(img_dir+'/*.tif') # list all files in the img folder
    num_examples = len(img_filenames)
    # 把样本数目写入在一个txt中，之后需要用
    with open(num_examples_txt_file,'w') as f:
        f.write(str(num_examples))

    for i in range(len(img_filenames)):
        img_name = img_filenames[i].split("\\")[1]
        # 对应的label文件
        label_filename = label_dir + '/' + img_name
        #img = Image.open(img_filenames[i])
        img = io.imread(img_filenames[i])
        #label = Image.open(label_filename)
        label = io.imread(label_filename)

        # transfer label from RGB to gray value
        im_size = label.shape[:2]
        new_mask = np.zeros(im_size,dtype='uint8')
        c1 = np.stack([label[...,0]<50,label[...,1]<50,label[...,2]>200],axis=-1)
        new_mask[c1.all(-1)] = 255
        #label = new_mask

        # resize image and label (for test in the trial)
        # in practice, we have to crop the images to avoid artifacts 
        img = resize(img,(256,256,3),preserve_range=True)
        new_mask = resize(new_mask,(256,256),preserve_range=True)
        img = img.astype('uint8')
        new_mask = new_mask.astype('uint8')
        # 将图像和Label转化为二进制格式
        img_raw = img.tobytes()
        label_raw = new_mask.tobytes()
        # 对label和img进行封装
        example = tf.train.Example(features=tf.train.Features(feature={
            "label_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        # 序列化为字符串
        writer.write(example.SerializeToString())

    writer.close()
    print("%s finished！"%(tfrecord_file))
    t1 = time.time()
    print("running time is %s s"%(str(t1-t0)))

if __name__=="__main__":
    # 训练样本文件夹
    base_dir = 'D:/Git_projects/Building-Extraction/data/vaihingen-data-batch/'
    train_img_dir = base_dir + 'training/image'
    train_label_dir = base_dir + 'training/label'

    # 验证样本文件夹
    val_img_dir = base_dir + 'validation/image'
    val_label_dir = base_dir + 'validation/label'

    # 生成的训练样本tfrecords文件
    train_tfrecords_file = base_dir + "training/train.tfrecords"
    train_num_examples_txt_file = base_dir + "training/train_num_examples.txt"
    img2TFRecord(train_img_dir,train_label_dir,train_tfrecords_file,train_num_examples_txt_file)
    # 生成的验证样本tfrecords文件
    val_tfrecords_file = base_dir + "validation/validation.tfrecords"
    val_num_examples_txt_file = base_dir + "validation/val_num_examples.txt"
    img2TFRecord(val_img_dir,val_label_dir,val_tfrecords_file,val_num_examples_txt_file)

