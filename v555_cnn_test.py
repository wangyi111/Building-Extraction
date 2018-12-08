
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import sys


# In[2]:

#tfrecord文件存放路径
TFRECORD_FILE="D:/jupyter_pycode/buildings/11111/test.tfrecords"

#预测结果存放路径
output_dir="D:/jupyter_pycode/buildings/00000trial/prediction_clipping/"



BATCH_SIZE=1

TOTAL_ITER=81000


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
    #获取图片数据
    
    
    
    image_order=features['order']
    image_name=features['name']
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
image_order,image_name,image,image_raw,label=read_and_decode(TFRECORD_FILE)  #tf文件     ??数量+格式？？？

##【v2】将标签转为一维
label=tf.reshape(label,[256]) 



##【v2】不能打乱了
#给训练样本分批次
image_order_batch,image_name_batch,image_batch,image_raw_batch,label_batch=tf.train.batch(
[image_order,image_name,image,image_raw,label],batch_size=BATCH_SIZE,capacity=5000,num_threads=1)  #参数设多少



##定义网络结构
with tf.Session() as sess:
    #初始化权值
    def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.01)  #生成一个截断的正态分布
        return tf.Variable(initial)
    
    #初始化偏置
    def bias_variable(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
    
    #卷积层
    def conv2d_L1(x,W):
        # x input tensor of shape: [batch_size,in_height,in_width,in_channels]
        # W filter / kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
        # strides[0]=strides[3]=1, strides[1]表示x方向步长，strides[2]表示y方向步长
        #padding: A string from: "SAME","VALID"
        return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')
    
    #池化层
    def max_pool_2x2(x):
        #ksize [1,x,y,1]
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
    
    
    def conv2d_L2(x,W):
        return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')
    
    def conv2d_L3(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
    def batch_normalize(x):
        axes=[d for d in range(len(x.get_shape()))]
        mean,var=tf.nn.moments(x,axes=axes)
        scale = tf.Variable(tf.constant(1.0,shape=[]))
        offset = tf.Variable(tf.constant(0.0,shape=[]))
        variance_epsilon = 0.00001
        x2 = tf.nn.batch_normalization(x, mean, var, offset, scale, variance_epsilon)       
        return x2
    
    #定义两个placeholoder
    x=tf.placeholder(tf.float32,[None,64,64,3])
    y=tf.placeholder(tf.int32,[None,256])
    
    
    #改变x的格式，转为4D向量[batch_size,in_height,in_width,in_channels]
    x_image=tf.reshape(x,[-1,64,64,3])
    
    #初始化第一个卷积层的权值和偏置
    W_conv1=weight_variable([9,9,3,64])  #16*16采样窗口，64个卷积核，从3个平面抽取特征，得到64个特征平面
    b_conv1=bias_variable([64])  #每一个卷积核有一个偏置值
    
    #把x_image和权值向量进行卷积，再加上偏置，然后应用于relu激活函数
    h_conv1=tf.nn.relu(conv2d_L1(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1)  #最大值池化
    
    #h_pool1=batch_normalize(h_pool1)
    
    #第二个卷积层
    W_conv2=weight_variable([7,7,64,128])  #112个卷积核从64个平面抽取特征
    b_conv2=bias_variable([128])
    
    h_conv2=tf.nn.relu(conv2d_L2(h_pool1,W_conv2)+b_conv2)
    #h_pool2=max_pool_2x2(h_conv2)
    h_pool2=h_conv2  #第二层不池化
    
    #h_pool2=batch_normalize(h_pool2)
    
    #第三个卷积层
    W_conv3=weight_variable([5,5,128,64])  #64个卷积核从32个平面抽取特征
    b_conv3=bias_variable([64])
    
    h_conv3=tf.nn.relu(conv2d_L3(h_pool2,W_conv3)+b_conv3)
    #h_pool2=max_pool_2x2(h_conv2)
    h_pool3=h_conv3  #第三层不池化
    
    #h_pool3=batch_normalize(h_pool3)
    
    #64*64的图片第一次卷积后还是32*32，第一次池化后变为32*32，第二次卷积后16*16，第三次卷积后16*16，得到64张16*16的特征平面
    
    

   
    #初始化第一个全连接层的权值
    W_fc1=weight_variable([16*16*64,4096])  #上一层输出64*64*80个神经元，全连接层有4096个神经元
    b_fc1=bias_variable([4096])
    
    #池化层2的输出扁平化为1维
    h_pool3_flat=tf.reshape(h_pool3,[-1,16*16*64])
    #求第一个全连接层的输出
    h_fc1=tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)
    
    #dropout层稍降维，keep_prob表示使用神经元的概率
    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    
    #h_fc1_drop=batch_normalize(h_fc1_drop)
    
    #初始化第二个全连接层
    W_fc2=weight_variable([4096,256*2])
    b_fc2=bias_variable([256*2])
     
        
    ###########要改
                                 
    #计算输出 
    net_output=tf.matmul(h_fc1_drop,W_fc2)+b_fc2  #256*1 概率
    
    #net_output=batch_normalize(net_output)
    
    
    
    logits=tf.reshape(net_output,[-1,256,2])

    prediction=tf.nn.softmax(logits)
    prediction2=tf.argmax(prediction,2)
    prediction2=tf.cast(prediction2,tf.int32)
    #【v2】二值化
    #prediction2=tf.round(prediction)  #取距离最近的整数（0.5以下--0；0.5以上--1）  Q：tensor格式中怎样自己设阈值
    
    #判断预测正确性，结果存放在bool型列表中                              
    correct_prediction=tf.equal(y,prediction2)  #对比，返回True或False；
    correct_prediction=tf.reshape(correct_prediction,[-1])
    #求准确率                              
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将bool型转为浮点型（t&f转为1&0）

    
    #保存模型
    saver=tf.train.Saver()
    #初始化
    sess.run(tf.global_variables_initializer())
    
    
    saver.restore(sess,'D:/jupyter_pycode/buildings/11111/v4_CNN_softmax_onehot_noBN_10x8100/trial_model.model')
        
    #创建一个协调器，管理线程
    coord=tf.train.Coordinator()
    #启动QueueRunner,此时文件名队列已经进队
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)    

    for num in range(TOTAL_ITER):
        #获取一个批次的数据和标签
        
        sys.stdout.write('\r>> Converting image %d/%d' % (num+1,TOTAL_ITER))
        sys.stdout.flush()
        
        
        
        
        b_image_order,b_image_name,b_image,b_image_raw,b_label=sess.run([image_order_batch,image_name_batch,
                                                                         image_batch,image_raw_batch,label_batch])
        
        b_label1=b_label.reshape(1,16,16)
        
        ##【v2】获取文件名、序列
        b_image_order=b_image_order.tolist()
        b_image_name=b_image_name.tolist()
        b_image_name_str=b_image_name[0].decode()
        
        

        
        ###测试tfrecords是否准确存储数据
        
        #print(b_image_raw.shape,b_image_raw[0].shape,b_label1.shape)
        
        '''
        #显示图像 灰度图
        img=Image.fromarray(b_image_raw[0],mode='RGB')

        plt.imshow(img)
        plt.axis('off')
        plt.show()
     
        #显示标签
        lab=Image.fromarray(b_label1[0])
        
        plt.imshow(lab)
        plt.axis('off')
        plt.show()
        '''
        
        
        
        ##【v2】预测出图   
        predict_array=sess.run(prediction,feed_dict={x:b_image,keep_prob:1.0})  
        #print(predict_array)
        predict_img=predict_array.reshape(1,16,16,2)
        '''
        pre_img=Image.fromarray(predict_img[0])
    
        #二值化
        pix = pre_img.load()
        
        width = pre_img.size[0]
        height = pre_img.size[1]
        '''
      
        img = Image.new('RGB', (16, 16), (0,0,0))
        draw = ImageDraw.Draw(img)
        for w in range(16):
            for h in range(16):
                r=predict_img[0,w,h]
                #rr=r/255
                if r[1]>0.5:
                    draw.point((w, h), fill=(255,0,0))
      
        
        #plt.imshow(img)
        #plt.axis('off')
        #plt.show()

        #print(b_label1[0])
        
        #print(sess.run(logits,feed_dict={x:b_image,keep_prob:1.0}))
        
    
        #print(sess.run(net_output,feed_dict={x:b_image,keep_prob:1.0}))
        
        
        #预测好的16*16标签存到本地文件
        
        
        outfile=os.path.join(output_dir,b_image_name_str + '.png')
        
        img.save(outfile)
        
        
        
        

    
        '''    
        #判断预测正确性，结果存放在bool型列表中                              
        correct_prediction=tf.equal(y,prediction2)  #对比，返回True或False；
        correct_prediction=tf.reshape(correct_prediction,[-1])
        #求准确率                              
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将bool型转为浮点型（t&f转为1&0）
        '''
        #acc=sess.run(accuracy,feed_dict={x:b_image,y:b_label,keep_prob:1.0})
        
        #print(b_image_name_str+'  '+str(acc))
                
    sys.stdout.write('\n')
    sys.stdout.flush()     
        
    #通知其他线程关闭
    coord.request_stop()
    #其他所有线程关闭后，返回函数
    coord.join(threads)
    
    
    
    
        

print('complete.')
    
    
    
    
    


