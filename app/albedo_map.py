from re import sub
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



import numpy as np
from lib.renderer.camera import Camera
import numpy as np
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
import cv2
import time
import math
import random
import pyexr
import argparse
from tqdm import tqdm
from lib.renderer.gl.prt_render import PRTRender
from lib.renderer.gl.render import Render
from lib.renderer.gl.cam_render import CamRender
from lib.renderer.gl.scrender import scRender

dataset_path="D:/workspace/SRTP/data/THuman2.0_new/0000"

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def render_sc(out_path, folder_name, subject_name,  rndr:PRTRender , im_size, angl_step=4, n_light=1, pitch=[0]):
    cam = Camera(width=im_size, height=im_size)
    cam.ortho_ratio = 0.4 * (512 / im_size)
    cam.near = -100
    cam.far = 100
    cam.sanity_check()
    mesh_file = os.path.join(folder_name, subject_name + '.obj')
    if not os.path.exists(mesh_file):
        print('ERROR: obj file does not exist!!', mesh_file)
        return
    prt_file = os.path.join(folder_name, 'bounce', 'bounce0.txt')
    if not os.path.exists(prt_file):
        print('ERROR: prt file does not exist!!!', prt_file)
        return
    face_prt_file = os.path.join(folder_name, 'bounce', 'face.npy')
    if not os.path.exists(face_prt_file):
        print('ERROR: face prt file does not exist!!!', prt_file)
        return
    text_file = os.path.join(folder_name,  'material0.jpeg')
    if not os.path.exists(text_file):
        print('ERROR: dif file does not exist!!', text_file)
        return
    
    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(
        mesh_file, with_normal=True, with_texture=True)
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2

    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])

    rndr.set_norm_mat(y_scale, vmed)
    tan, bitan = compute_tangent(
        vertices, faces, normals, textures, face_textures)
    rndr.set_mesh(vertices, faces, normals, faces_normals,
                  textures, face_textures,  tan, bitan)
    rndr.set_albedo(texture_image)
    
    os.makedirs(os.path.join(out_path, 'GEO',
                'OBJ', subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'PARAM', subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'RENDER', subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'MASK', subject_name), exist_ok=True)


    for p in pitch:
        for y in tqdm(range(0, 360, angl_step)):
            R = np.matmul(make_rotate(math.radians(p), 0, 0),
                          make_rotate(0, math.radians(y), 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90), 0, 0))

            rndr.rot_matrix = R
            rndr.set_camera(cam)

            for j in range(n_light):

                dic = {'ortho_ratio': cam.ortho_ratio,
                       'scale': y_scale, 'center': vmed, 'R': R}

                rndr.display()

                out_all_f = rndr.get_color(0)
                out_mask = out_all_f[:, :, 3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)

                np.save(os.path.join(out_path, 'PARAM', subject_name,
                        '%d_%d_%02d.npy' % (y, p, j)), dic)
                cv2.imwrite(os.path.join(out_path, 'RENDER', subject_name,
                            '%d_%d_%02d.jpg' % (y, p, j)), 255.0*out_all_f)
                cv2.imwrite(os.path.join(out_path, 'MASK', subject_name,
                            '%d_%d_%02d.png' % (y, p, j)), 255.0*out_mask)

                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default=dataset_path)
    parser.add_argument('-o', '--out_dir', type=str,
                        default='./data')
    parser.add_argument('-m', '--ms_rate', type=int, default=1,
                        help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  action='store_true',
                        help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int,
                        default=512, help='rendering image size')
    args = parser.parse_args()
    from lib.renderer.gl.init_gl import initialize_GL_context
    initialize_GL_context(width=args.size, height=args.size, egl=args.egl)

    from lib.renderer.gl.scrender import scRender
    rndr=scRender(args.size, args.size, ms_rate=args.ms_rate, egl=args.egl)
    
    if args.input[-1] == '/':
        args.input = args.input[:-1]
    subject_name = args.input.split('/')[-1]
    
    render_sc(args.out_dir, args.input, subject_name,
                      rndr,args.size, 1, 1, pitch=[0])
    