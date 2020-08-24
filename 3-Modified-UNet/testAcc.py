import numpy as np
import glob
import os
from skimage import io

def testAccuracy(test_label,test_pred,testAccuracyDir):
    """
    计算所有测试图像的精度，精度评定指标为PA,IOU,precision,recall,F1_score并把每张图像的精度写到一个txt中
    :param test_label:测试图像标签所在的文件夹
    :param test_pred:测试图像的预测结果所在的文件夹
    :param testAccuracyDir:存放精度结果的文件夹
    :return:None
    """
    TP = 0  # 真真值，预测为1，真值为1
    TN = 0  # 真假值，预测为0，真值为0
    FP = 0  # 假真值，预测为1,真值为0
    FN = 0  # 假假值，预测为0，真值为1
    testlabeldir=glob.glob(test_label+'/*.tif')
    for i in range(len(testlabeldir)):
        testlabel=io.imread(testlabeldir[i])
        # print(testlabel.shape)
        testpreddir=test_pred+'/'+testlabeldir[i].split('\\')[1]
        testpred=io.imread(testpreddir)
        # print(testpred.shape)
        row,col=testlabel.shape
        for m in range(row):
            for n in range(col):
                if testlabel[m,n]==1 and testpred[m,n]==1:
                    TP+=1
                if testlabel[m,n]==0 and testpred[m,n]==0:
                    TN+=1
                if testlabel[m,n]==0 and testpred[m,n]==1:
                    FP+=1
                if testlabel[m,n]==1 and testpred[m,n]==0:
                    FN+=1
    # 所有测试图像的精度
    PA=(TP+TN)/(row*col)
    IoU=TP/(FN+TP+FP)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1_score=(2*precision*recall)/(precision+recall)
    #将结果写入txt，
    with open(testAccuracyDir+'/'+'testAccuracy.txt',"w") as f:
        f.write('PA='+str(PA)+'\n'+'IoU='+str(IoU)+'\n'+'precision='+str(precision)+'\n'+'recall='+str(recall)+'\n')
    #将结果打印出来
    print("测试集精度为：")
    print('PA='+str(PA)+'\n'+'IoU='+str(IoU)+'\n'+'precision='+str(precision)+'\n'+'recall='+str(recall)+'\n')

if __name__=="__main__":
    test_label='./mini_data/val/segs'
    test_pred='./mini_data/predict_val'
    testAccuracyDir='./mini_data/valAccuracy'

    if not os.path.exists(testAccuracyDir):
        os.mkdir(testAccuracyDir)

    testAccuracy(test_label,test_pred,testAccuracyDir)

