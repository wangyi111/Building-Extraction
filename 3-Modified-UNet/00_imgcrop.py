import pdb
from skimage import io
import numpy as np
import math
import glob
import os
import time
import shutil

#判断RGB图像白色的比例
def white_proportion_of_img(img):
    row,col = img.shape[0:2]
    #print(img.shape[0:2])
    white_num = 0.0
    for i in range(row):
        for j in range(col):
            if img[i,j,0]==255 and img[i,j,1]==255 and img[i,j,2]==255:
                white_num += 1
    white_proportion = white_num/np.float32(row*col)

    return  white_proportion

def crop(imgs_in, segs_in, imgs_out, segs_out, size, stride):
    num = 0
    filename = glob.glob(imgs_in+'/*.tiff')
    for i in range(len(filename)):
        print("准备裁剪第%d张影像..."%(i+1))
        img = io.imread(imgs_in + '/' + filename[i].split('\\')[1])
        #seg = io.imread(segs_in + filename[i].split('\\')[1])
        seg = np.loadtxt(segs_in + '/' + filename[i].split('\\')[1].split('.')[0] + '.txt')
        seg = np.uint8(seg)
        length, width = img.shape[0:2]
        m = math.floor((length - size)/stride) + 1
        n = math.floor((width - size)/stride) + 1
        for j in range(m):
            for k in range(n):               
                img_patch = img[stride*j:stride*j+size, stride*k:stride*k+size, :]
                seg_patch = seg[stride*j:stride*j+size, stride*k:stride*k+size]
                # 影像块中白色部分的比例
                white_proportion = white_proportion_of_img(img_patch)
                print("white proportion of img_patch:%f"%(white_proportion))
                if white_proportion<0.01:
                    io.imsave(imgs_out + '/' + str(num) + '.tif', img_patch)
                    io.imsave(segs_out + '/' + str(num) + '.tif', seg_patch)
                    num += 1
    print("共得到裁剪瓦片数目为:%d"%(num))


if __name__ == '__main__':
    t0 = time.time()
    print("开始裁剪验证样本..")
    # 验证样本
    base_dir = 'E:/profshao/毕业交接/发表的论文以及程序/BRRNet程序代码/Mini_DataSet/'
    val_imgs_in = base_dir + 'ValidationSet/InputImages'
    val_segs_in = base_dir + 'ValidationSet/TargetMaps_txt'
    val_imgs_out = base_dir + 'ValidationSet/imgs'
    val_segs_out = base_dir + 'ValidationSet/segs'
    #pdb.set_trace()1
    if os.path.exists(val_imgs_out):shutil.rmtree(val_imgs_out)
    if os.path.exists(val_segs_out): shutil.rmtree(val_segs_out)
    if not os.path.exists(val_imgs_out): os.mkdir(val_imgs_out)
    if not os.path.exists(val_segs_out): os.mkdir(val_segs_out)

    size = 256
    stride = 64
    crop(val_imgs_in, val_segs_in, val_imgs_out, val_segs_out, size, stride)
    print("裁剪验证样本完成！")

    print("开始裁剪训练样本..")
    # 训练样本
    train_imgs_in = base_dir + 'TrainingSet/InputImages'
    train_segs_in = base_dir + 'TrainingSet/TargetMaps_txt'
    train_imgs_out = base_dir + 'TrainingSet/imgs'
    train_segs_out = base_dir + 'TrainingSet/segs'
    #pdb.set_trace()
    if os.path.exists(train_imgs_out):shutil.rmtree(train_imgs_out)
    if os.path.exists(train_segs_out): shutil.rmtree(train_segs_out)
    if not os.path.exists(train_imgs_out): os.mkdir(train_imgs_out)
    if not os.path.exists(train_segs_out): os.mkdir(train_segs_out)

    size = 256
    stride = 64
    crop(train_imgs_in, train_segs_in, train_imgs_out, train_segs_out, size, stride)
    print("裁剪训练样本完成！")

    print("开始裁剪测试样本..")
    # 测试样本
    test_imgs_in = base_dir + 'TestSet/InputImages'
    test_segs_in = base_dir + 'TestSet/TargetMaps_txt'
    test_imgs_out = base_dir + 'TestSet/imgs'
    test_segs_out = base_dir + 'TestSet/segs'
    #pdb.set_trace()
    if os.path.exists(test_imgs_out):shutil.rmtree(test_imgs_out)
    if os.path.exists(test_segs_out): shutil.rmtree(test_segs_out)
    if not os.path.exists(test_imgs_out): os.mkdir(test_imgs_out)
    if not os.path.exists(test_segs_out): os.mkdir(test_segs_out)

    size = 256
    stride = 64
    crop(test_imgs_in, test_segs_in, test_imgs_out, test_segs_out, size, stride)
    print("测试训练样本完成！")

    t1 = time.time()
    print("总共用时为：%s s"%(str(t1-t0)))