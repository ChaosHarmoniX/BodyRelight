## turn spherical harmonics into a image

# load sh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import OpenEXR
import Imath
import os
import scipy
import pyshtools as pysh
import cv2
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def filter1channel(oneChannel,sh0,sh1):
    clm = pysh.SHGrid.from_array(oneChannel)
    clm = clm.expand()
    # topo = pysh.expand.MakeGridDH(clm.coeffs, sampling=2) / 1000.
    # coeffs = pysh.expand.SHExpandDH(topo, sampling=2)
    # coeffs_filtered = coeffs.copy()
    coeffs_filtered = clm.coeffs
    # print("coeffs shape is ",clm.coeffs.shape)
    coeffs_filtered[:, :sh0, :] = 0.
    coeffs_filtered[:, sh1:, :] = 0.
    coef= coeffs_filtered[:, sh0:sh1, :]    
    # print(coef.shape)
    # get the (1,8) array of the coefficients
    
    
    
    final_coef = coef[0,:,0]/10
    print(final_coef)
    
    return final_coef



def image2sh(image,sh0,sh1):
    img=image
    r_img = img[:, :, 0]
    g_img = img[:, :, 1]
    b_img = img[:, :, 2]
    rc=filter1channel(r_img,sh0,sh1)
    gc=filter1channel(g_img,sh0,sh1)
    bc=filter1channel(b_img,sh0,sh1)
    # make 3 (9,) array to (1,9,3) array
    sh=np.array([rc,gc,bc])
    sh=sh.transpose(1,0)
    sh=np.expand_dims(sh,axis=0)
    
    
    return sh

# def render_exr(exrpath:str,target_folder:str,target_name:str):
#     File = OpenEXR.InputFile(exrpath)
#     PixType = Imath.PixelType(Imath.PixelType.FLOAT)
#     DW = File.header()['dataWindow']
#     Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
#     rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
#     mytiff = np.zeros((Size[1], Size[0]), dtype=np.float32)
#     rgb = np.array(rgb)
#     rgb = rgb.reshape(3, Size[1], Size[0])
#     rgb = rgb.transpose(1, 2, 0)
#     rgb=np.clip(rgb,0,1)
#     rgb=np.power(rgb,0.25)
#     rgb=np.clip(rgb,0,1)
#     rgb = (rgb * 255).astype(np.uint8)
#     target_path=os.path.join(target_folder,target_name+".jpg")
#     cv2.imwrite(target_path,rgb)
    
def exr2img(exrpath:str):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    mytiff = np.zeros((Size[1], Size[0]), dtype=np.float32)
    rgb = np.array(rgb)
    rgb = rgb.reshape(3, Size[1], Size[0])
    rgb = rgb.transpose(1, 2, 0)
    rgb=np.clip(rgb,0,1)
    rgb=np.power(rgb,0.25)
    rgb=np.clip(rgb,0,1)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb



img = exr2img('datas/s/2.exr')
img = cv2.resize(img,(1024,512))
cv2.imshow('img',img)
cv2.waitKey(0)
ssss
print("read exr done")  
# resize
# imgshow=cv2.resize(img,(512,256))
# cv2.imshow('img',imgshow)
# cv2.waitKey(0)

sh=image2sh(img,0,9)
print("sh done")
print(sh)
# save as new.npy
np.save('env_sh.npy',sh)


# read sh
shs = np.load('datas/sh/env_sh.npy')
print(shs[0])

    
    
    
    


# sh=np.load('datas/sh/env_sh.npy')
# print(sh.shape)
# # 240张图像
# # 9个球谐函数
# # 3个通道

# # use the function from scipy
# from scipy.special import sph_harm
# # 建立球面

# for image_index in range(240):
#     image_sphere=np.zeros((360,180,3))
#     for i in range(360):
#         for j in range(180):
#             for channel in range(3):
#                 cnt=0
#                 for l in range(2):
#                     for m in range(2*l+1):
#                         image_sphere[i][j][channel]+=sh[image_index][cnt][channel]*sph_harm(m-l,l,float(i)/180*np.pi,float(j)/180*np.pi).real
#                         cnt+=1
#     plt.imshow(image_sphere)
#     plt.savefig('datas/shres/sh'+str(image_index)+'.png')









