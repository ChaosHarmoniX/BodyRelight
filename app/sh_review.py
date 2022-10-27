## turn spherical harmonics into a image

# load sh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy





sh=np.load('datas/sh/env_sh.npy')
print(sh.shape)
# 240张图像
# 9个球谐函数
# 3个通道

# use the function from scipy
from scipy.special import sph_harm
# 建立球面

for image_index in range(240):
    image_sphere=np.zeros((360,180,3))
    for i in range(360):
        for j in range(180):
            for channel in range(3):
                cnt=0
                for l in range(2):
                    for m in range(2*l+1):
                        image_sphere[i][j][channel]+=sh[image_index][cnt][channel]*sph_harm(m-l,l,float(i)/180*np.pi,float(j)/180*np.pi).real
                        cnt+=1
    plt.imshow(image_sphere)
    plt.savefig('datas/shres/sh'+str(image_index)+'.png')









