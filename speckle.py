import numpy as np
import math
from PIL import Image
from scipy.linalg import expm
import cv2 as cv

'''
Num为散斑颗粒数，Height和Width为图像大小，R为散斑直径
'''
Num = 1500
Height = 128
Width = 128
I0 = 1
R = 3


def ceate_speckle():
    I = np.zeros((Height, Width), dtype=float)
    for i in range(Height):
        for j in range(Width):
            xk = np.random.randint(1, Num, [1, Height])
            yk = np.random.randint(1, Num, [1, Width])
            Y = I0*(expm(-((np.dot((i-xk).T, i-xk))+(np.dot((j-yk).T, j-yk)))/(R ^ 2)))
            I[i][j] = Y.sum()
    max_intensity = I.max()
    min_intensity = I.min()
    normalized_I = (I-min_intensity)/(max_intensity-min_intensity)
    img = 255*normalized_I
    img = img.astype(np.uint8)
    cv.imshow('speckle image', img)
    cv.waitKey(0)
    cv.imwrite('speckle.jpg', img)


def create_affine():
    original = cv.imread('speckle.jpg')
    rows, cols, ch = original.shape
    #平移矩阵
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
    #旋转矩阵
    pts2 = np.float32([[cols*0.2, rows*0.1], [cols*0.9, rows*0.2], [cols*0.1, rows*0.9]])

    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(original, M, (cols, rows))

    cv.imshow('image', dst)
    k = cv.waitKey(0)
    if k == ord('s'):
        cv.imwrite('affine_speckle.jpg', dst)
        cv.destroyAllWindows()


#读出的图像为三通道
if __name__ == '__main__':
    

