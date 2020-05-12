import cv2
import numpy as np
import glob
from os.path import join, basename, splitext, isdir
import os
import random
from PIL import Image

#  mask=cv2.imread('lisa-cnn-attack/masks/mask_l1loss_uniform_rectangles.png')[:,:,0]
img_list=[i for i in glob.glob(join('lisa-cnn-attack/optimization_output/l1basedmask_uniformrectangles/noisy_images','*400.png'))]

def _imwrite(dst_file,img):
    return img.save(dst_file)

def _inpaint(img_list, mask, dst_file):
    for im in img_list:
        im_name=splitext(basename(im))[0]
        img=cv2.imread(im)
        inpainted_img = cv2.inpaint(img,mask,3, cv2.INPAINT_TELEA)
        cv2.imwrite(dst_file + im_name + '.png',inpainted_img)

def randomizeMask(k,w, mask_size=32):
    mask_arr=np.zeros(shape=(mask_size,mask_size))
    mask_in=mask_size-w+1
    w_ind_x=list()
    w_ind_y=list()
    for i in xrange(k):
        w_ind_x=w_ind_x+ [random.randint(0,mask_in)]
        w_ind_y=w_ind_y+[random.randint(0,mask_in)]
    for j in zip(w_ind_x,w_ind_y):
        mask_arr[j[0]:j[0]+w,j[1]:j[1]+w]=1
    return mask_arr

def _resizeImg(img_src, img_dst):
    img=cv2.resize(cv2.imread(img_src),(32,32))
    return cv2.imwrite(img_dst,img)

def num_size_window(k_list=[2,3,4,5,6,7,8,9,10], w_list=[4,5,6,7,8,9,10]):
    kw=[]
    for k in k_list:
        for w in w_list:
            kw=kw+[(k,w)]
    print( kw)

# c=0
# for i in kw:
#     dst_folder='inpaint_TELEA'+str(c)
#     if not isdir(join('lisa-cnn-attack', dst_folder)):
#         os.mkdir(join('lisa-cnn-attack', dst_folder))
#     for j in xrange(1000):
#         maskarr=randomizeMask(i[0],i[1])
#         mask=Image.fromarray(np.uint8(255*maskarr))
#         _imwrite(join('lisa-cnn-attack',dst_folder,str(j)+'_mask.png'),mask)
#         mask=cv2.imread(join('lisa-cnn-attack',dst_folder,str(j)+'_mask.png'))[:,:,0]
#         _inpaint(img_list, mask, join('lisa-cnn-attack',dst_folder,str(j)+'_'))
#
#     c=c+1

# img=cv2.imread('/home/xinwwei/Desktop/original_result (4).png')
# print img.shape
# img=cv2.resize(img,(32,32))
# cv2.imwrite('/home/xinwwei/Desktop/resize(4).png',img)
# print img.shape

crop_img=[i for i in glob.glob("Crop-2998.jpg")]
print(len(crop_img))
for img_src in crop_img:
    im_name = splitext(basename(img_src))[0]
    img_dst="new/cropResized.png"
    _resizeImg(img_src,img_dst)
