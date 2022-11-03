'''
    Import this from PIFu: https://github.com/shunsukesaito/PIFu
    Usage:
        指定数据的路径 根据与计算出的结果 结合SH参数 生成渲染数据
        由于生成数据使用的是PRT方法 所以需要先执行prt_util.py中的方法
'''




from genericpath import exists
from re import sub
import sys
import os

from matplotlib import test

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#####
##
###
##





from lib.renderer.camera import Camera
import numpy as np
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
import cv2
import time
import math
import random
import pyexr
import argparse
import trimesh
from tqdm import tqdm
from lib.renderer.gl.prt_render import PRTRender
from lib.renderer.gl.render import Render
from lib.renderer.gl.cam_render import CamRender
from lib.renderer.gl.scrender import scRender

import app.prt_util
from app.prt_util import testPRT
from app.albedo_map import render_sc
from app.transfer_map import render_transfer_map
from app.render_data import render_prt_ortho

data_root_PATH=ROOT_PATH+'/data'
raw_data_path="D:/workspace/SRTP/data/THuman2.0_new/"
sh_src_npy_path=ROOT_PATH+'/datas/sh/env_sh.npy'

def gen_all_datasset():
    shs=np.load(sh_src_npy_path)
    data_folders=os.listdir(raw_data_path)
    
    print("Root Path: "+ROOT_PATH)
    print("SHS shape: "+str(shs.shape))
    
    if(not os.path.exists(os.path.join(ROOT_PATH, 'data'))):
        os.mkdir(os.path.join(ROOT_PATH, 'data'))
        
    # arguments:
    arg_size=512
    arg_ms_rate=1
    arg_egl=False
    from lib.renderer.gl.init_gl import initialize_GL_context
    initialize_GL_context(width=arg_size, height=arg_size, egl=arg_egl)
    
    for folder_of_raw in data_folders:
        # folder_of_raw='0001'
        print("Processing: "+folder_of_raw)
        os.makedirs(os.path.join(data_root_PATH,folder_of_raw),exist_ok=True)
        obj_root_path=os.path.join(data_root_PATH,folder_of_raw)# data folder
        obj_src_path=os.path.join(raw_data_path,folder_of_raw)# obj file path
        # create folder in obj_root_path
        os.makedirs(os.path.join(obj_root_path,'ALBEDO'),exist_ok=True)
        os.makedirs(os.path.join(obj_root_path,'TRANSFORM'),exist_ok=True)
        os.makedirs(os.path.join(obj_root_path,'IMAGE'),exist_ok=True)
        # calculate the PRT
        # check if the bounce file exists
        if(not os.path.exists(os.path.join(obj_src_path,'bounce/face.npy'))):
            print("Calculating PRT...")
            testPRT(obj_src_path)
            print("Done")
        else:
            print("PRT file exist")
            
        # generate ALBEDO
        if(not os.path.exists(os.path.join(obj_root_path,'ALBEDO','ALBEDO.jpg'))):
            print("ALBEDO NOT FOUND")
            albedo_rndr=scRender(arg_size,arg_size,ms_rate=arg_ms_rate,egl=arg_egl)
            render_sc(obj_root_path, obj_src_path, folder_of_raw, albedo_rndr, arg_size, arg_ms_rate, arg_egl, pitch=[0])
        else:
            print("ALBEDO file exist")
        # generate TRANSFORM
        if(not os.path.exists(os.path.join(obj_root_path,'TRANSFORM','8.jpg'))):
            print("TRANSFORM NOT FOUND")
            rndr=PRTRender(arg_size,arg_size,ms_rate=arg_ms_rate,egl=arg_egl)
            render_transfer_map(obj_root_path, obj_src_path, folder_of_raw, shs, rndr, arg_size, 1, 9, pitch=[0])
        else:
            print("TRANSFORM file exist")
        # generate IMAGE
        if(not os.path.exists(os.path.join(obj_root_path,'IMAGE','IMAGE.jpg'))):
            print("IMAGE NOT FOUND")
            rndr=PRTRender(arg_size,arg_size,ms_rate=arg_ms_rate,egl=arg_egl)
            render_prt_ortho(obj_root_path, obj_src_path, folder_of_raw, shs, rndr,  arg_size,  pitch=[0])
        else:
            print("IMAGE file exist")
              
            

             
        
        
        
        
        
    
    
    
    
    

        
        
    
    



if __name__ == '__main__':
    gen_all_datasset()