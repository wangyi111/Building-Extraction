import tensorflow as tf
import glob
import numpy as np
import os
from skimage import io

########################以下是将图像创建为dataset，但是只适用于jpeg,gif,png,bmp四种格式，tif格式的创建见下面get_dataset_from_tfrecords##################
def read_img_label_from_filename(img_filename, label_filename):
    """
    从给定的图像和标签的路径读取对应文件
    :param img_filename: 图像的路径
    :param label_filename:  标签的路径
    :return: 图像，标签
    """
    img_str = tf.read_file(img_filename)
    img = tf.image.decode_image(img_str)
    label_str = tf.read_file(label_filename)
    label = tf.image.decode_image(label_str)

    return img, label

def get_dataset_from_img(img_dir,label_dir,threads=1,batch_size=1,shuffle=True):
    """
    创建dataset
    :param img_dir: 存放图像的文件夹
    :param label_dir: 存放标签的文件夹
    :param threads: 处理时的线程数目
    :param batch_size: 批次大小
    :param shuffle: 是否随机打乱dataset中的元素
    :return: dataset
    """
    img_filenames =glob.glob(img_dir+'/*.jpeg')
    label_filenames = glob.glob(label_dir+'/*.jpeg')
    num_img = len(img_filenames)
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices((img_filenames, label_filenames))
        # 把每个(img_filename, label_filename)处理成(img,label)
        dataset = dataset.map(read_img_label_from_filename, num_parallel_calls=threads)

        if shuffle:
            dataset = dataset.shuffle(num_img)

        # 分批次并重复dataset
        dataset = dataset.batch(batch_size).repeat()

    return dataset

########################使用tfrecords文件创建dataset##############################################
img_shape = (256,256,3)

# 解析TFRecorddataset
def parse_function(example_proto):
  features = {"img_raw": tf.io.FixedLenFeature((), tf.string),
              "label_raw": tf.io.FixedLenFeature((), tf.string)}
  parsed_features = tf.io.parse_single_example(example_proto, features)
  img_str = parsed_features["img_raw"]
  label_str = parsed_features["label_raw"]
  # 对 tf.string进行解码
  img = tf.io.decode_raw(img_str,tf.uint8)
  img = tf.reshape(img,img_shape)

  label = tf.io.decode_raw(label_str,tf.uint8)
  label = tf.reshape(label,img_shape[:2])

  # 一个label在keras中要求有3-D
  label = tf.expand_dims(label, axis=-1)

  # 归一化
  img = tf.cast(img,tf.float32) * (1.0/255)
  label = tf.cast(label,tf.float32)
  # 注意当label取 0,255时需要去掉下面的注释；当label取 0,1时不需要
  # label = label  * (1.0/255)

  return img, label

def get_dataset_from_tfrecords(tfrecords_file,num_examples,threads=1,batch_size=1,shuffle=True):
    """
    使用tfrecords文件创建dataset
    :tfrecords_file:
    :threads:
    :batch_size:
    :shuffle:
    """

    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset.map(parse_function, num_parallel_calls=threads)
    # shuffle data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_examples)
    # batch data
    dataset = dataset.batch(batch_size).repeat()
    # prefetch data
    dataset.prefetch(num_examples // batch_size)
        
    return dataset

if __name__=="__main__":
    # tfrecords文件路径
    base_dir = 'D:/Git_projects/Building-Extraction/data/vaihingen-data-batch/'
    train_tfrecords_file = base_dir + "training/train.tfrecords"
    val_tfrecords_file = base_dir + "validation/validation.tfrecords"
    train_num_examples_txt_file = base_dir +"training/train_num_examples.txt"
    val_num_examples_txt_file = base_dir + "validation/val_num_examples.txt"

    # 读取样本数目
    with open(train_num_examples_txt_file, 'r') as f:
        train_num_examples = np.int(f.readline())
    with open(val_num_examples_txt_file, 'r') as f:
        val_num_examples = np.int(f.readline())
    print("train_num_examples：%d"%(train_num_examples))
    print("val_num_examples：%d" % (val_num_examples))

    print("开始创建train_dataset和val_dataset...")
    train_dataset = get_dataset_from_tfrecords(train_tfrecords_file, train_num_examples,threads=1,batch_size=2,shuffle=True)
    val_dataset = get_dataset_from_tfrecords(val_tfrecords_file,val_num_examples,threads=1,batch_size=2,shuffle=True)
    print("创建train_dataset和val_dataset完成！")

    import matplotlib.pyplot as plt
    plt.figure()
    for i,(img,label) in enumerate(train_dataset.unbatch().take(2)):
        ax = plt.subplot(2,2,2*i+1)
        ax.imshow(img.numpy())
        bx = plt.subplot(2,2,2*i+2)
        bx.imshow(label[...,0].numpy()*255)

    plt.show()