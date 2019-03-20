# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random

# imsize=128
# filepath='E:/UnderWaterDataset/Zebrafish/resize'+str(imsize)+'/'
# data=[]
# for filename in os.listdir(filepath):
#     # print (os.path.join(filepath,filename))
#     imagename=os.path.join(filepath,filename)
#     img = cv2.imread(imagename)
#     data.append(img)
# np.save('E:/UnderWaterDataset/Zebrafish_'+str(imsize)+'.npy',data)
# print('Data has saved ! ')

# 将多个类别的数据合并在一起，并制作标签
def mulClassData(imSize, shffle=True):
    X01 = np.load('E:/UnderWaterDataset/Coral_' + str(imSize) + '.npy')
    X02 = np.load('E:/UnderWaterDataset/Jellyfish_' + str(imSize) + '.npy')
    X03 = np.load('E:/UnderWaterDataset/Parrotfish_' + str(imSize) + '.npy')
    X04 = np.load('E:/UnderWaterDataset/SeaAnemone_' + str(imSize) + '.npy')
    X05 = np.load('E:/UnderWaterDataset/Zebrafish_' + str(imSize) + '.npy')
    data = np.concatenate([X01, X02, X03, X04, X05], axis=0)
    label = np.concatenate([np.zeros(X01.shape[0]), np.zeros(X02.shape[0]) + 1, np.zeros(X03.shape[0]) + 2,
                           np.zeros(X04.shape[0]) + 3, np.zeros(X05.shape[0]) + 4], axis=0)
    if shffle:
        permutation=np.random.permutation(data.shape[0])
        data=data[permutation,:,:]
        label=label[permutation]
    np.savez('E:/UnderWaterDataset/mergeData_'+str(imSize)+'.npz',data,label)
        # 另一种打乱数据集的方式
        # index = [i for i in range(len(data))]
        # random.shuffle(index)
        # data = data[index]
        # label = label[index]
    return 0

if __name__ == '__main__':
    imSize = 128
    mulClassData(imSize)
