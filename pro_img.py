import time
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from estimator import *





model_path = "/Users/sedecim/Downloads/multi_person_mobilenet_v1_075_float.tflite"
estimator = Estimator(16,model_path,0.5,5,50,1)
cam = cv2.VideoCapture(0)

while True:

    ret_val, image = cam.read()

    image = crop_image(image)
    (fac,fitted_img) = resize_img(image)

    poses = estimator.process_img(fitted_img,fac)

    estimator.draw_poses(poses,image)

    cv2.imshow('cam_demo',image)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()