# -*- coding: utf-8 -*-
# image = cv2.imread("C:/Users/shou/Desktop/1.jpg")
# # Flipped Horizontally 水平翻转
# h_flip = cv2.flip(image, 1)
# cv2.imwrite("C:/Users/shou/Desktop/1-h.jpg", h_flip)

from PIL import Image,ImageChops
from PIL import ImageEnhance
import os
import cv2
import numpy as np

# 平移， 平移后右边少的部分又补在了左边，不太好
def move(root_path,img_name,xoff,yoff): #平移，平移尺度为xoff,yoff
    img = Image.open(os.path.join(root_path, img_name))
    offset = ImageChops.offset(img,xoff,yoff)
    return offset

def cropImage(root_path, img_name,w0,h0):   # w0,h0分别是开始裁剪的x,和y 的位置
    img=Image.open(os.path.join(root_path,img_name))
    cutImg=img.crop((w0,h0,img.size[0]-20,img.size[1]-20))
    return cutImg
# 翻转
def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

# 旋转
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(20) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

# 随机颜色
def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

# 对比度增强
def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

# 亮度增强
def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


# 颜色增强
def colorEnhancement(root_path,img_name):#颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored

if __name__=='__main__':
    imageDir="E:/UnderWaterDataset/Coral/" #要改变的图片的路径文件夹
    saveDir="E:/UnderWaterDataset/Coral/save/"   #要保存的图片的路径文件夹
    file_list=os.listdir(imageDir)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    i=0
    for name in file_list:
        # saveName1="flip_"+str(i)+".png"
        # flip_saveImage=flip(imageDir,name)
        # flip_saveImage.save(os.path.join(saveDir,saveName1))
        #
        # saveName2 = "colEnhan_" + str(i) + ".png"
        # colEnhan_saveImage = colorEnhancement(imageDir, name)
        # colEnhan_saveImage.save(os.path.join(saveDir, saveName2))

        saveName4 = "crop_" + str(i) + ".png"
        crop_saveImage = cropImage(imageDir, name,20,20)
        crop_saveImage.save(os.path.join(saveDir, saveName4))

        # saveName5 = "conEnhan_" + str(i) + ".png"
        # conEnhan_saveImage = contrastEnhancement(imageDir, name)
        # conEnhan_saveImage.save(os.path.join(saveDir, saveName5))
        #
        # saveName6 = "briEnhan_" + str(i) + ".png"
        # brihan_saveImage = brightnessEnhancement(imageDir, name)
        # brihan_saveImage.save(os.path.join(saveDir, saveName6))

        # saveName3 = "rotation_" + str(i) + ".png"
        # rotation_saveImage=rotation(imageDir,name)
        # rotation_saveImage.save(os.path.join(saveDir,saveName3))

        # saveName7 = "randCo_" + str(i) + ".png"
        # randCo__saveImage = randomColor(imageDir, name)
        # randCo__saveImage.save(os.path.join(saveDir, saveName7))
        i=i+1
    print('data augmentation is finished !')

