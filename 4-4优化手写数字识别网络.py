
# coding: utf-8

# In[2]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

#载入数据集，自动下载
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)  #one_hot:标签转为0、1形式


#每个批次的大小
batch_size=100
#计算批次数
n_batch=mnist.train.num_examples // batch_size

#定义placeholder
x=tf.placeholder(tf.float32,[None,784])  #像素
y=tf.placeholder(tf.float32,[None,10])  #标签
keep_prob=tf.placeholder(tf.float32)  #dropout参数
lr=tf.Variable(0.001,dtype=tf.float32)  #学习率

#创建神经网络
W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))  #初始化优化
b1=tf.Variable(tf.zeros([500])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)  #keep_prob设置工作神经元数目

W2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([300,100],stddev=0.1))
b3=tf.Variable(tf.zeros([100])+0.1)
L3=tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop=tf.nn.dropout(L3,keep_prob)

W4=tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)

prediction=tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)  #softmax:输出信号转为概率


#二次代价函数:适合线性神经元
# loss=tf.reduce_mean(tf.square(y-prediction))

#对数释然代价函数:适合softmax函数输出
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#Adam下降法优化
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#判断预测正确性，结果存放在bool型列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #对比，返回True或False；argmax返回最大值所在位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将bool型转为浮点型（t&f转为1&0）

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))  #每次训练调整学习率（慢慢降低）
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        #测试准确率    
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter "+ str(epoch) + "Testing Accuracy " + str(test_acc) + "Training Accuracy" + str(train_acc))
        
        
#改进方法：批次大小、增加隐藏层、权&偏置的初始化、代价函数、优化函数、学习率、训练次数


# In[ ]:



