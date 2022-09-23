import OpenEXR
import Imath
import numpy as np
import cv2
import imageio
import os


def render_exr(exrpath:str,target_folder:str,target_name:str)-> None:
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    mytiff = np.zeros((Size[1], Size[0]), dtype=np.float32)
    rgb = np.array(rgb)
    rgb = rgb.reshape(3, Size[1], Size[0])
    rgb = rgb.transpose(1, 2, 0)
    rgb=np.clip(np.power(rgb,1/2.2),0,1)

    rgb = (rgb * 255).astype(np.uint8)
    target_path=os.path.join(target_folder,target_name+".jpg")
    cv2.imwrite(target_path,rgb)

def render_exr_another_way(exrpath:str)-> None:
    base_num=2**16-1
    print(base_num)
    im=cv2.imread(exrpath,-1)
    im=im*base_num
    im[im>base_num]=base_num
    im=np.uint16(im)
    cv2.imwrite("./output/sc/"+exrpath[:-4]+".png",im)
