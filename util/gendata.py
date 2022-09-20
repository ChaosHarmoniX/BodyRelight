'''
    Generate datas for Body_Relight
'''
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.renderer.camera import Camera
import numpy as np
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl


'''
    Generate png for a mesh at the default view
'''
def generate_png(mesh: str, png_path: str):
    # load mesh
    mesh = load_obj_mesh_mtl(mesh)