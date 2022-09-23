''' 
    EXR转JPG
'''
# datasets_folder：你的数据集文件夹
datasets_folder="D:\\Computer Programing\\SRTP\\datas\\100samplesDataset"
# output_folder：你的输出文件夹
output_folder="./output"
#
import os
import sys
import tqdm
# add the path of the util folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util.exr import render_exr

def get_exr_paths(exr_dir):
    exr_paths = []
    for root, dirs, files in os.walk(exr_dir):
        for file in files:
            if file.endswith(".exr"):
                exr_paths.append(os.path.join(root, file))
    return exr_paths

if __name__ == '__main__':
    exr_paths = get_exr_paths(datasets_folder)
    cnt = 0
    for exr_path in exr_paths:
        cnt += 1
        render_exr(exr_path, output_folder, str(cnt))
        if cnt > 10:
            break
