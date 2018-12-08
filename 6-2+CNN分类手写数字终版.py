
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size=100
n_batch=mnist.train.num_examples // batch_size



#初始化权值
def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.1)  #生成一个截断的正态分布
    return tf.Variable(initial,name=name)

#初始化偏置
def bias_variable(shape,name):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

#定义卷积
def conv2d(x,W):
    # x input tensor of shape: [batch_size,in_height,in_width,in_channels]
    # W filter / kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
    # strides[0]=strides[3]=1, strides[1]表示x方向步长，strides[2]表示y方向步长
    #padding: A string from: "SAME","VALID"
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义池化
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


with tf.name_scope('input'):
    
    #定义两个placeholoder
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
    
    with tf.name_scope('x_image'):
        #改变x的格式，转为4D向量[batch_size,in_height,in_width,in_channels]
        x_image=tf.reshape(x,[-1,28,28,1],name='x_image')

with tf.name_scope('conv1_pool1'):
    #初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1=weight_variable([5,5,1,32],name='W_conv1')  #5*5采样窗口，32个卷积核，从一个平面抽取特征，得到32个特征平面
    with tf.name_scope('b_conv1'):
        b_conv1=bias_variable([32],name='b_conv1')  #每一个卷积核有一个偏置值

    #把x_image和权值向量进行卷积，再加上偏置，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1=conv2d(x_image,W_conv1)+b_conv1
    with tf.name_scope('relu'):    
        h_conv1=tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1=max_pool_2x2(h_conv1)  #最大值池化
        
with tf.name_scope('conv2_pool2'):   
    #第二个卷积层和池化层
    with tf.name_scope('W_conv2'):
        W_conv2=weight_variable([5,5,32,64],name='W_conv2')  #64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2=bias_variable([64],name='b_conv2')
    with tf.name_scope('conv2d_2'):
        conv2d_2=conv2d(h_pool1,W_conv2)+b_conv2
    with tf.name_scope('relu'):    
        h_conv2=tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2=max_pool_2x2(h_conv2)

#28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14，第二次卷积后14*14，第二次池化后7*7，得到64张7*7的特征平面

with tf.name_scope('fc1'):
    #初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1=weight_variable([7*7*64,1024],name='W_fc1')  #上一层输出7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1=bias_variable([1024],name='b_fc1')

    with tf.name_scope('h_pool2_flat'):
        #池化层2的输出扁平化为1维
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64],name='h_pool2_flat')
    with tf.name_scope('wx_plus_b1'):
        #求第一个全连接层的输出
        wx_plus_b1=tf.matmul(h_pool2_flat,W_fc1)+b_fc1
    with tf.name_scope('relu'):
        h_fc1=tf.nn.relu(wx_plus_b1)

    with tf.name_scope('keep_prob'):
        #dropout层稍降维，keep_prob表示使用神经元的概率
        keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob,name='h_fc1_drop')

with tf.name_scope('fc2'):
    #初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2=weight_variable([1024,10],name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2=bias_variable([10],name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    with tf.name_scope('softmax'):
        #计算输出
        prediction=tf.nn.softmax(wx_plus_b2)

with tf.name_scope('cross_entropy'):
    #对数释然代价函数:适合softmax函数输出
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
    
with tf.name_scope('train'):
    #Adam下降法优化
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #判断预测正确性，结果存放在bool型列表中
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #对比，返回True或False；argmax返回最大值所在位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将bool型转为浮点型（t&f转为1&0）
        tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #初始化变量
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)
    
    for i in range(1001):
        #训练模型
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        
        #记录训练集参数
        summary=sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        train_writer.add_summary(summary,i)
        #记录测试集参数
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        summary=sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        test_writer.add_summary(summary,i)
        
        if i%100==0: 
            test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            train_acc=sess.run(accuracy,feed_dict={x:mnist.test.images[:10000],y:mnist.test.labels[:10000],keep_prob:1.0})
            print("Iter "+ str(i) + "Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))


# In[ ]:



