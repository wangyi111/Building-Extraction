
# coding: utf-8

# In[1]:

import tensorflow as tf
from nets import resnet_v2 
import os
from PIL import Image,ImageDraw
import numpy as np
import time

import sys

start=time.clock()

#批次?
BATCH_SIZE=1


#学习率
learn_rate=5e-5

TOTAL_ITER=22448

#tfrecord文件存放路径
TFRECORD_FILE="D:/jupyter_pycode/buildings/33333/train.tfrecords"

#预测结果存放路径
output_dir="D:/jupyter_pycode/buildings/33333/prediction_clipping/"


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
    image=tf.cast(image,tf.float32)
    #image=tf.subtract(image,0.5)
    #image=tf.multiply(image,2.0)
    
    #获取标签数据
    label=tf.decode_raw(features['label'],tf.uint8)
    label=tf.reshape(label,[16,16])
       
    return image_order,image_name,image,image_raw,label


##获取图片数据和标签
image_order,image_name,image,image_raw,label=read_and_decode(TFRECORD_FILE)  #tf文件


##将标签转为一维
label=tf.reshape(label,[256]) 




#给训练样本分批次
image_order_batch,image_name_batch,image_batch,image_raw_batch,label_batch=tf.train.batch(
[image_order,image_name,image,image_raw,label],batch_size=BATCH_SIZE,capacity=5000,num_threads=1)  #参数设多少



##定义网络结构

with tf.Session() as sess:
    #定义两个placeholoder
    x=tf.placeholder(tf.float32,[None,64,64,3])
    y=tf.placeholder(tf.int32,[None,256])
    
    
    X=tf.reshape(x,[BATCH_SIZE,64,64,3])
    net, end_points = resnet_v2.resnet_v2_50(X,is_training=False,global_pool=False)  #得到最后一个block的输出
    
    #print(net.shape)
    
    
    ##全连接层
    
    #初始化权值
    def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.01)  #生成一个截断的正态分布
        return tf.Variable(initial)
    
    #初始化偏置
    def bias_variable(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    #初始化第一个全连接层的权值
    W_fc1=weight_variable([2*2*2048,256*2])  #上一层输出7*7*64个神经元，全连接层有1024个神经元
    b_fc1=bias_variable([256*2])
    
    #池化层2的输出扁平化为1维
    net_flat=tf.reshape(net,[-1,2*2*2048])
    
    #dropout层稍降维，keep_prob表示使用神经元的概率
    keep_prob=tf.placeholder(tf.float32)
    net_drop=tf.nn.dropout(net_flat,keep_prob)
    
    
    
    #求第一个全连接层的输出
    net_output=tf.matmul(net_drop,W_fc1)+b_fc1
    
    logits=tf.reshape(net_output,[-1,256,2])
    
    
    with tf.name_scope('cross_entropy'):
        #交叉熵代价函数                    
        cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits),name='cross_entropy')   
        tf.summary.scalar('cross_entropy',cross_entropy)
        
    #Momentum优化                            
    train_step=tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy) 
    
    
    prediction=tf.nn.softmax(logits)
    
    prediction2=tf.argmax(prediction,2)
    prediction2=tf.cast(prediction2,tf.int32)
    #判断预测正确性，结果存放在bool型列表中                              
    correct_prediction=tf.equal(y,prediction2)  #对比，返回True或False；
    correct_prediction=tf.reshape(correct_prediction,[-1])
    
    with tf.name_scope('accuracy'):
        #求准确率                              
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将bool型转为浮点型（t&f转为1&0）
        tf.summary.scalar('accuracy',accuracy)
        
    merged=tf.summary.merge_all()

    
    #保存模型
    saver=tf.train.Saver()
    #初始化
    sess.run(tf.global_variables_initializer())
    
    writer=tf.summary.FileWriter('logs_resnet/',sess.graph)  #存储图的结构
    
    
    saver.restore(sess,'D:/jupyter_pycode/buildings/33333/model/trial_model.model')
    
    
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
                #pp=r[1]*255
                #pp=int(pp)
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

end=time.clock()
print(end-start,' s')



