
import numpy as np
import math
import cv2
import os
import sys

cubemapFaceNormals = [
    [[1, 0, 0], [0, 0, -1], [0, -1, 0]],  # posx
    [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],  # negx
    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],  # posy
    [[0, -1, 0], [1, 0, 0], [0, 0, -1]],  # negy
    [[0, 0, 1], [1, 0, 0], [0, -1, 0]],  # posz
    [[0, 0, -1], [-1, 0, 0], [0, -1, 0]]  # negz
]
# faces is a 6*1 list, each element is a image
def main(faces, ch):
    size = faces[0].shape[0]
    channels = ch 
    cubeMapVecs = []
    
    for index in range(len(faces)):
        faceVecs = []
        for v in range(size):
            for u in range(size):
                fU = (u + 0.5) / size
                fV = (v + 0.5) / size
                
                vecX = np.array(cubemapFaceNormals[index][0])
                vecX = vecX * fU
                
                VecY = np.array(cubemapFaceNormals[index][1])
                vecY = VecY * fV
                
                VecZ = np.array(cubemapFaceNormals[index][2])
                
                res = vecX + vecY + VecZ
                
                res = res / np.linalg.norm(res)
                
                faceVecs.append(res)
        cubeMapVecs.append(faceVecs)
    
    sh = [
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32),
        np.zeros((3), dtype=np.float32)
    ]
    
    weightAccum = 0.0
    
    for index in range(len(faces)):
        pixels = faces[index].reshape(-1)
        gammaCorrect = False
        for y in range(size):
            for x in range(size):
                texelVect = cubeMapVecs[index][y*size+x]
                weight = texelSolidAngle(x, y, size, size)
                
                weight1 = weight * 4.0 / 17.0
                weight2 = weight * 8.0 / 17.0
                weight3 = weight * 15.0 / 17.0
                weight4 = weight * 5.0 / 68.0
                weight5 = weight * 15.0 / 68.0
                
                dx = texelVect[0]
                dy = texelVect[1]
                dz = texelVect[2]
                
                for c in range(3):
                    value = pixels[(y*size+x)*3+c]/255.0
                    if gammaCorrect:
                        value = value ** 2.2
                    
                    sh[0][c] += value * weight1
                    sh[1][c] += value * weight2 * dx
                    sh[2][c] += value * weight2 * dy
                    sh[3][c] += value * weight2 * dz
                    sh[4][c] += value * weight3 * dx * dz
                    sh[5][c] += value * weight3 * dz * dy
                    sh[6][c] += value * weight3 * dy * dx
                    sh[7][c] += value * weight4 * (3.0 * dz * dz - 1.0)
                    sh[8][c] += value * weight5 * (dx * dx - dy * dy)
                    
                    weightAccum += weight
                    
                    
    for i in range(len(sh)):
        sh[i][0] *= 4 * math.pi / weightAccum
        sh[i][1] *= 4 * math.pi / weightAccum
        sh[i][2] *= 4 * math.pi / weightAccum
    
    return sh

def texelSolidAngle(aU, aV, width, height):
    U = (2.0 * (aU + 0.5) / width) - 1.0
    V = (2.0 * (aV + 0.5) / height) - 1.0
    
    invResolutionW = 1.0 / width
    invResolutionH = 1.0 / height
    
    x0 = U - invResolutionW
    y0 = V - invResolutionH
    x1 = U + invResolutionW
    y1 = V + invResolutionH
    
    angle = areaElement(x0, y0) - areaElement(x0, y1) - areaElement(x1, y0) + areaElement(x1, y1)
    
    return angle

def areaElement(x, y):
    return math.atan2(x * y, math.sqrt(x * x + y * y + 1.0))

CubeMapFiles = [
    "./CubeMap/RT_Backward.hdr",
    "./CubeMap/RT_Down.hdr",
    "./CubeMap/RT_Forward.hdr",
    "./CubeMap/RT_Left.hdr",
    "./CubeMap/RT_Right.hdr",
    "./CubeMap/RT_Up.hdr"
]



if __name__=="__main__":
    # abstract cubemap from hdr
    faces = []
    for i in range(len(CubeMapFiles)):
        faces.append(cv2.imread(CubeMapFiles[i], cv2.IMREAD_UNCHANGED)*255)
        # convert to png
        cv2.imwrite("face"+str(i)+".png", faces[i])
        
        # cv2.imshow("face", faces[i])
        # cv2.waitKey(0)
    sh = main(faces, 9)
    # turn sh to 9*3 matrix
    sh = np.array(sh)
    sh = sh.reshape(9,3)
    # dump sh 
    np.save("sh.npy", sh)
    print(sh)      
                
        
                
                        
        
    



    