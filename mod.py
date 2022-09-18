'''
    A local Trial Script to test the functions
'''

# public packages
import trimesh as tm
import os


# private packages
from util import gendata
from lib.renderer.camera import *
from lib.renderer.mesh import *
from lib.renderer.gl.render import *

dataset_path="D:/Computer Programing/SRTP/datas/THuman2.0_new/"

def get_meshes_pathes(dataset_path: str):
    meshes_pathes=[]
    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".obj"):
                meshes_pathes.append(os.path.join(root,file))
    return meshes_pathes

if __name__ == "__main__":
    meshes_pathes=get_meshes_pathes(dataset_path)
    # for mesh_path in meshes_pathes:
    #     mesh=tm.load(mesh_path)
    #     print(mesh_path)
    #     print(mesh)

    # Get mesh vertices and faces
    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(
        meshes_pathes[0], with_normal=True, with_texture=True)
    from lib.renderer.gl.prt_render import PRTRender
    rndr = PRTRender(width=1600, height=1200, ms_rate=1, egl=True)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures)
    rndr.draw() # 一直gl error
            


    