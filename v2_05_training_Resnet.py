
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
from PIL import Image
#from nets import nets_factory
import numpy as np
import time
#from tensorflow.contrib.slim.nets import resnet_v1
#from nets import resnet_v1
import collections # 原生的collections库
import my_resnet


# In[2]:

start=time.clock()

#批次?
BATCH_SIZE=10

#CHAR_SET_LEN=256


#学习率

#tfrecord文件存放路径
TFRECORD_FILE="D:/jupyter_pycode/buildings/00000trial/train.tfrecords"


##函数：从tfrecord读取数据
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue=tf.train.string_input_producer([filename])
    reader=tf.TFRecordReader()
    #返回文件名和文件
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,
                                     features={
                                         'order':tf.FixedLenFeature([],tf.int64),
                                         'name':tf.FixedLenFeature([],tf.string),
                                         'image':tf.FixedLenFeature([],tf.string),
                                         'label':tf.FixedLenFeature([],tf.string),
                                     
        })
    #tf.train.shuffle_batch必须确定shape
    
    image_order=features['order']
    image_name=features['name']
    
    #获取图片数据
    image=tf.decode_raw(features['image'],tf.uint8)    
    #未处理图
       
    image_raw=tf.reshape(image,[64,64,3])
    #图片预处理
    image=tf.reshape(image,[64,64,3])
    image=tf.cast(image,tf.float32) / 255.0
    image=tf.subtract(image,0.5)
    image=tf.multiply(image,2.0)
    
    #获取标签数据
    label=tf.decode_raw(features['label'],tf.uint8)
    label=tf.reshape(label,[16,16])
       
    return image_order,image_name,image,image_raw,label


##获取图片数据和标签
image_order,image_name,image,image_raw,label=read_and_decode(TFRECORD_FILE)  #tf文件


##将标签转为一维
label=tf.reshape(label,[256]) 


#给训练样本分批次
image_order_batch,image_name_batch,image_batch,image_raw_batch,label_batch=tf.train.shuffle_batch(
[image_order,image_name,image,image_raw,label],batch_size=BATCH_SIZE,capacity=5000,min_after_dequeue=1000,num_threads=1)  #参数设多少


##定义网络结构


slim = tf.contrib.slim # 使用方便的contrib.slim库来辅助创建ResNet

with tf.Session() as sess:
    

    
        #定义两个placeholoder
    x=tf.placeholder(tf.float32,[None,64,64,3])
    y=tf.placeholder(tf.float32,[None,256])
        
    X=tf.reshape(x,[BATCH_SIZE,64,64,3])
        
    with slim.arg_scope(my_resnet.resnet_arg_scope(is_training=False)): # is_training设置为false
        net, end_points = my_resnet.resnet_v2_50(X, None)  #[batch_size,1,1,2048]
    
    #初始化权值
    def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.1)  #生成一个截断的正态分布
        return tf.Variable(initial)
    
    #初始化偏置
    def bias_variable(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
    
    

    
    #初始化第一个全连接层的权值
    W_fc1=weight_variable([2048,256])  #上一层输出7*7*64个神经元，全连接层有1024个神经元
    b_fc1=bias_variable([256])
    
    #池化层2的输出扁平化为1维
    net_flat=tf.reshape(net,[-1,1*1*2048])
    
    #dropout层稍降维，keep_prob表示使用神经元的概率
    keep_prob=tf.placeholder(tf.float32)
    net_drop=tf.nn.dropout(net_flat,keep_prob)
    
    #求第一个全连接层的输出
    prediction=tf.nn.softmax(tf.matmul(net_drop,W_fc1)+b_fc1)

    
    
    #二值化
    prediction2=tf.round(prediction)  #取距离最近的整数（0.5以下--0；0.5以上--1）  Q：tensor格式中怎样自己设阈值

    
    
    #设阈值0.5，二值化
    
    
                                 
    #对数释然代价函数，怎么改？                    
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))                               
    #Adam下降法优化                            
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
                                    
    ##需另改判断方法    
    #判断预测正确性，结果存放在bool型列表中                              
    correct_prediction=tf.equal(y,prediction2)  #对比，返回True或False；
    correct_prediction=tf.reshape(correct_prediction,[-1])
    #求准确率                              
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将bool型转为浮点型（t&f转为1&0）
        
    

    
    #保存模型
    saver=tf.train.Saver()
    #初始化
    sess.run(tf.global_variables_initializer())
    
    
    #创建一个协调器，管理线程
    coord=tf.train.Coordinator()
    #启动QueueRunner,此时文件名队列已经进队
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    
    for i in range(11):
        
        #获取一个批次的数据和标签
        b_image,b_label=sess.run([image_batch,label_batch])
        
        loss,_=sess.run([cross_entropy,train_step],feed_dict={x:b_image,y:b_label,keep_prob:0.7})
        
        if i%2==0:
            
            acc=sess.run(accuracy,feed_dict={x:b_image,y:b_label,keep_prob:1.0})           
            #print("Iter "+ str(i) + "accuracy= " + str(acc))
            
            print("Iter "+str(i)+" loss= "+str(loss)+" accuray= "+str(acc))
        
        '''
        if i%200==0:
            show_predict=sess.run(prediction,feed_dict={x:b_image,keep_prob:0.7})
            print(show_predict)
        '''
        
            #if acc>0.9:
    #saver.save(sess,"D:/jupyter_pycode/buildings/00000trial/trial_model.model")
                #break


    #通知其他线程关闭
    coord.request_stop()
    #其他所有线程关闭后，返回函数
    coord.join(threads)
    
    



print('complete.')

end=time.clock()
print(end-start,' s')








