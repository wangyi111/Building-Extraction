import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from create_Dataset import *

class UNet:
    """
    建立BRRNet网络
    """
    def __init__(self,img_shape,num_classes,log_dir,savemodel_path,device='/gpu:0'):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.log_dir = log_dir
        self.savemodel_path = savemodel_path
        self.device = device

    ############模型结构###############
    def conv_block(self,input_tensor,num_filters):
        encoder = layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self,input_tensor, num_filters):
        encoder = self.conv_block(input_tensor=input_tensor, num_filters=num_filters)
        encoder_pool = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(encoder)
        return encoder, encoder_pool

    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(filters=num_filters, kernel_size=(2,2), strides=(2,2),padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor,decoder],axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = self.conv_block(input_tensor=decoder, num_filters=num_filters)
        return decoder

    ####################更改center结构##################
    def center_block(self, input_tensor, num_filters):
        center_1 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',dilation_rate=1)(input_tensor)
        center_1 = layers.BatchNormalization()(center_1)
        center_1 = layers.Activation('relu')(center_1)

        center_2 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(center_1)
        center_2 = layers.BatchNormalization()(center_2)
        center_2 = layers.Activation('relu')(center_2)

        center_3 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=4)(center_2)
        center_3 = layers.BatchNormalization()(center_3)
        center_3 = layers.Activation('relu')(center_3)

        center_4 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=8)(center_3)
        center_4 = layers.BatchNormalization()(center_4)
        center_4 = layers.Activation('relu')(center_4)

        center_5 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=16)(center_4)
        center_5 = layers.BatchNormalization()(center_5)
        center_5 = layers.Activation('relu')(center_5)

        center_6 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=32)(center_5)
        center_6 = layers.BatchNormalization()(center_6)
        center_6 = layers.Activation('relu')(center_6)

        center = layers.Add()([center_1, center_2, center_3, center_4, center_5, center_6])

        return center
    ######################################################

        ###########################残差修正模块###############################
    # 加入残差修正模块，该模块也是使用空洞卷积串联的结构，最后加1个3x3的卷积将通道数变为1，然后把之前的output_1也add进来，最后接sigmoid
    def rrm_ours(self, input_tensor, num_filters):
        x_1 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=1)(input_tensor)
        x_1 = layers.BatchNormalization()(x_1)
        x_1 = layers.Activation('relu')(x_1)

        x_2 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(x_1)
        x_2 = layers.BatchNormalization()(x_2)
        x_2 = layers.Activation('relu')(x_2)

        x_3 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=4)(x_2)
        x_3 = layers.BatchNormalization()(x_3)
        x_3 = layers.Activation('relu')(x_3)

        x_4 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=8)(x_3)
        x_4 = layers.BatchNormalization()(x_4)
        x_4 = layers.Activation('relu')(x_4)

        x_5 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=16)(x_4)
        x_5 = layers.BatchNormalization()(x_5)
        x_5 = layers.Activation('relu')(x_5)

        x_6 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=32)(x_5)
        x_6 = layers.BatchNormalization()(x_6)
        x_6 = layers.Activation('relu')(x_6)

        x = layers.Add()([x_1, x_2, x_3, x_4, x_5, x_6])

        x = layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        x = layers.Add()([input_tensor, x])

        output = layers.Activation('sigmoid')(x)

        return output
    

    #############自定义损失函数#####################
    def dice_coeff(self,y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    def dice_loss(self,y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    ######################################################
    
    def build(self):
        with tf.device(self.device):
            inputs = layers.Input(shape=self.img_shape)
            # 图像大小 256
            encoder_1,encoder_pool_1 = self.encoder_block(input_tensor=inputs,num_filters=64)
            # 128
            encoder_2,encoder_pool_2 = self.encoder_block(input_tensor=encoder_pool_1, num_filters=128)
            # 64
            encoder_3,encoder_pool_3 = self.encoder_block(input_tensor=encoder_pool_2, num_filters=256)
            # 32

            # center = self.conv_block(input_tensor=encoder_pool_4,num_filters=1024)
            center = self.center_block(input_tensor=encoder_pool_3, num_filters=512)

            decoder_3 = self.decoder_block(input_tensor=center, concat_tensor=encoder_3, num_filters=256)
            # 64
            decoder_2 = self.decoder_block(input_tensor=decoder_3, concat_tensor=encoder_2, num_filters=128)
            # 128
            decoder_1 = self.decoder_block(input_tensor=decoder_2, concat_tensor=encoder_1, num_filters=64)
            # 256

            outputs_1 = layers.Conv2D(filters=1,kernel_size=(1,1),activation='sigmoid')(decoder_1)

            # 加入rrm_ours进行单输出监督
            outputs_2 = self.rrm_ours(input_tensor=outputs_1, num_filters=64)

            model = keras.models.Model(inputs,outputs_2)

            # 打印出模型结构
            #print("model summary:")
            #model.summary()

            model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                          loss=self.dice_loss,
                          metrics=[keras.metrics.binary_accuracy]
                          )

        return model

    def train(self,epochs,train_dataset,val_dataset,num_train_examples,num_val_examples,learning_rate,batch_size):
        """
        训练
        :param epochs: 周期数
        :param log_dir: 保存日志的文件夹
        :param train_dataset: 训练数据集 dataset
        :param val_dataset:  验证数据集 dataset
        :param num_train_examples: 训练数据数目
        :param num_val_examples: 验证数据数目
        :return: history对象
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        with tf.device(self.device):
            model = self.build()
            # 加载保存的最好模型
            # if os.path.exists(self.savemodel_path):
            #     model = keras.models.load_model(self.savemodel_path)
            callbacks =[
                # 选最best_model 使用val_loss
                keras.callbacks.ModelCheckpoint(filepath=self.savemodel_path,monitor="val_loss",verbose=1,save_best_only=True,period=1),
                keras.callbacks.TensorBoard(log_dir=self.log_dir,batch_size=self.batch_size),
                # 使用动态衰减学习率
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1,mode='auto',min_delta=1e-4,cooldown=0,min_lr=1e-7)
            ]

            self.history = model.fit(train_dataset,
                                     epochs=self.epochs,
                                     callbacks=callbacks,
                                     validation_data=val_dataset,
                                     steps_per_epoch = int(np.ceil(num_train_examples / float(batch_size))),
                                     validation_steps=int(np.ceil(num_val_examples / float(batch_size))))
        return self.history

    def show_training_process(self):
        """
        展示训练过程中的loss和val_loss
        :return:
        """
        # train loss
        loss = self.history.history['loss']
        # val loss
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)
        # plt.figure(figsize=(16, 8))
        # plt.subplot(1, 2, 1)
        # plt.plot(epochs_range, dice, label='Training Dice Loss')
        # plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
        # plt.legend(loc='upper right')
        # plt.title('Training and Validation Dice Loss')

        #plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.savefig('./loss.jpg')
        #plt.show()

    def predict(self,img_dir,save_dir):
        # 加载训练好的模型
        model = keras.models.load_model(self.savemodel_path,custom_objects={'dice_loss':self.dice_loss})
        filenames = glob.glob(img_dir+'/*.tif')
        for i in range(len(filenames)):
            img_name = filenames[i].split("\\")[1]
            img = io.imread(filenames[i])
            # 归一化
            img = np.float32(img) * (1.0/255)
            #print(img.shape)
            img = np.expand_dims(img,0)
            #print(img.shape)
            prediction = model.predict(img)
            prediction[prediction>0.5] = 1
            prediction[prediction<=0.5] = 0
            # 把4-d 改为2-d
            prediction = prediction[0,:,:,0]
            #print('prediction shape')
            #print(prediction.shape)
            io.imsave(save_dir+'/'+img_name,prediction)
        print("预测结果保存路径：%s"%(save_dir))






