import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from estimator import *
import xlwt
from judge import vectorize


stride = 16
threshold = 0.5
nmsr = 20
radius = 1
detection = 3
model_path = "/Users/sedecim/Downloads/multi_person_mobilenet_v1_075_float.tflite"

estimator = Estimator(stride,model_path,threshold,detection,nmsr,radius)

# standard_file = "ikun/kun_24.png"

# def getPose(filename):
#     image = cv2.imread(filename)

#     iamge = crop_image(image)
#     (fac,fitted_img) = resize_img(image)
#     poses = estimator.process_img(fitted_img, fac)
#     main_pose = max(poses)
#     return main_pose

# std_pose = getPose(standard_file)

def pose_vectorize(pose):
    # return [1,2,3,4,5,6,7,8]
    return vectorize(pose)


def process_image(filename):
    image = cv2.imread(filename)

    iamge = crop_image(image)
    (fac,fitted_img) = resize_img(image)
    poses = estimator.process_img(fitted_img, fac)

    if(len(poses) > 0):
        main_pose = max(poses)
    
        vec = pose_vectorize(main_pose);
        return vec
    else:
        print((filename,"detect no pose"))
        return []

def write_observation(sheet ,line, vec):
    for i in range(len(vec)):
        sheet.write(line, i, vec[i])
    # sheet.write(line, len(vec), tag)


workbook = xlwt.Workbook()
sheet = workbook.add_sheet('sheet 1')

for i in range(100):
    filename = "kun_" + str(i) + ".png"
    vec = process_image(filename)
    write_observation(sheet, i, vec)


workbook.save('test.xls')