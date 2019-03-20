# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
fullfilename=[]
filepath="E:/UnderWaterDataset/Zebrafish/"
filepath1="E:/UnderWaterDataset/Zebrafish/resize32/"
if not os.path.exists(filepath):
    os.mkdir(filepath)

imsize=32
for filename in os.listdir(filepath):
    print (filename)
    print (os.path.join(filepath,filename))
    filelist=os.path.join(filepath,filename)
    fullfilename.append(filelist)
i=1
for imagename in fullfilename:
    img=cv2.imread(imagename)
    img=cv2.resize(img,(imsize,imsize))
    resizename=str(i)+'.png'
    isExists = os.path.exists(filepath1)
    if not isExists:
        os.makedirs(filepath1)
        print('mkdir resizename accomploshed')
    savename=filepath1+'/'+resizename
    cv2.imwrite(savename,img)
    print('{} is resized'.format(savename))
    i=i+1
