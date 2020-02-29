import muDIC as dic
import logging
import cv2 as cv
import numpy as np
from scipy import misc

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Path to folder containing images
#path = r'./example_data/' # Use this formatting on Linux and Mac OS
path = r'F:\GraduateProject\Python\PCA\speckle\\'
# path = r'c:\path\to\example_data\\'  # Use this formatting on Windows

# Generate image instance containing all images found in the folder
# images = dic.IO.image_stack_from_folder(path, file_type='.tif')
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
images.set_filter(dic.filtering.lowpass_gaussian, sigma=1.)


# Generate mesh
# 这里Xc1，Xc2，Yc1,Yc2是ROI的大小，需要根据输入图象的大小修改
mesher = dic.Mesher(deg_e=3, deg_n=3)
# mesh = mesher.mesh(images,Xc1=200,Xc2=1050,Yc1=200,Yc2=650,n_ely=8,n_elx=8, GUI=True)
mesh = mesher.mesh(images, Xc1=20, Xc2=105, Yc1=20, Yc2=65, n_ely=8, n_elx=8, GUI=True)


# Instantiate settings object and set some settings manually
settings = dic.DICInput(mesh, images)
# 图像数量
settings.max_nr_im = 2
settings.ref_update = [15]
settings.maxit = 20
# If you want to access the residual fields after the analysis, this should be set to True
settings.store_internals = False

# Instantiate job object
job = dic.DICAnalysis(settings)

# Running DIC analysis
dic_results = job.run()

# 用于仿真，假设z均为3,拼接出三维obj_points
# 世界三维坐标为参考图的点
cols, rows = dic_results.xnodesT.shape
obj_points = np.zeros((cols, 3), np.float32)
obj_points[:, 0] = dic_results.xnodesT[:, 0]
obj_points[:, 1] = dic_results.ynodesT[:, 0]
# obj_points = np.append(np.array(dic_results.xnodesT[:, 1]), np.array(dic_results.ynodesT[:, 1]))
znodes = np.zeros(cols, np.float32)
# obj_points = np.append(obj_points, znodes, axis=1)
obj_points[:, 2] = znodes
objectpoints = []
objectpoints.append(obj_points)

# img_points为扭曲图的点
img_points = np.zeros((cols, 2), np.float32)
img_points[:, 0] = dic_results.xnodesT[:, 1]
img_points[:, 1] = dic_results.ynodesT[:, 1]
imagepoints = []
imagepoints.append(img_points)
# img_points = np.append(dic_results.xnodesT[:, 0], dic_results.ynodesT[:, 0], axis=1)
# 得到size
weight, height = images[0].shape


# ret：？
# mtx:内参矩阵
# dist:畸变系数
# rvecs:旋转向量
# tvecs:平移向量

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectpoints, imagepoints, (weight, height), None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

# Calculate field values
fields = dic.post.viz.Fields(dic_results)

# Show a field
# viz = dic.Visualizer(fields, images=images)

# Uncomment the line below to see the results 显示最后一张
# viz.show(field="true strain", component=(1, 1), frame=39)

