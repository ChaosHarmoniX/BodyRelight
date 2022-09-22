import OpenEXR
import Imath
import numpy as np
import cv2


# 把exr文件路径放在这儿
filename="9C4A0006-5133111e97.exr"




def render_exr(exrpath:str)-> None:
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    mytiff = np.zeros((Size[1], Size[0]), dtype=np.float32)
    # generate rgb image to save the hdr image
    rgb = np.array(rgb)
    rgb = rgb.reshape(3, Size[1], Size[0])
    rgb = rgb.transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    cv2.imwrite(filename[:-4]+".png", rgb)



render_exr(filename)
