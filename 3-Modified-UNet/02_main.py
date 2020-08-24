#!/usr/bin/env python3

import os
import time
import shutil

from create_Dataset import *
from model_UNet_124681632_bn import *
from evaluate import *


""" os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) """

if __name__=="__main__":
    # 注意，每次修改完img_shape，都要修改create_Dataset.py中的img_shape一致
    img_shape = (256,256,3)

    # 指定使用的设备
    device = '/cpu:0'

    num_classes = 2
    batch_size = 2 #8
    epochs = 10 # 50
    learning_rate = 1e-3

    threads = 1 # 5 # 生成dataset的线程数

    log_dir = "D:/Git_projects/Building-Extraction/data/vaihingen-data-batch/log/"
    savemodel_dictionary = "D:/Git_projects/Building-Extraction/data/vaihingen-data-batch/model/"
    savemodel_path = "D:/Git_projects/Building-Extraction/data/vaihingen-data-batch/model/unet.hdf5"

    if not os.path.exists(log_dir):os.mkdir(log_dir)
    if not os.path.exists(savemodel_dictionary):os.mkdir(savemodel_dictionary)

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

    ##############################################创建训练样本和验证样本dataset#########################################
    print("开始创建train_dataset和val_dataset...")
    train_dataset = get_dataset_from_tfrecords(train_tfrecords_file, train_num_examples,threads=threads,batch_size=batch_size,shuffle=True)
    val_dataset = get_dataset_from_tfrecords(val_tfrecords_file,val_num_examples,threads=threads,batch_size=batch_size,shuffle=True)
    print("创建train_dataset和val_dataset完成！")

    ##############################################创建模型#########################################
    model = UNet(img_shape=img_shape, num_classes=num_classes, log_dir=log_dir, savemodel_path=savemodel_path,device=device)

    ##############################################训练#########################################
    t0 = time.time()
    model.train(epochs=epochs,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_train_examples=train_num_examples,
                num_val_examples=val_num_examples,
                learning_rate=learning_rate,
                batch_size=batch_size)
    t1 =time.time()
    print("训练用时为：%s s"%(str(t1-t0)))
    #可视化训练过程
    model.show_training_process()

    ##############################################测试#########################################
    img_dir = base_dir + "TestSet/imgs/"
    predict_save_dir = "Mini_DataSet/predict_test/"

    print("开始预测。带预测图像路径：%s"%(img_dir))
    if os.path.exists(predict_save_dir):shutil.rmtree(predict_save_dir)
    if not os.path.exists(predict_save_dir):os.mkdir(predict_save_dir)

    model.predict(img_dir=img_dir,save_dir=predict_save_dir)
    print("预测完成！")

    ##############################################精度评价#########################################
    test_label= base_dir + "TestSet/segs"
    test_pred= predict_save_dir
    testAccuracyDir= 'Mini_DataSet/testAccuracy'
    print("开始精度评价...")

    if not os.path.exists(testAccuracyDir):
        os.mkdir(testAccuracyDir)

    testAccuracy(test_label,test_pred,testAccuracyDir)
    print("精度评价完成！")















