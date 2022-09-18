# a local trial script
import trimesh as tm
import os

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
    for mesh_path in meshes_pathes:
        mesh=tm.load(mesh_path)
        print(mesh_path)
        print(mesh)