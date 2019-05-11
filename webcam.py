import time
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from estimator import *
from filtering import *




if __name__ == '__main__':

    model_path = "/Users/sedecim/Downloads/multi_person_mobilenet_v1_075_float.tflite"

    stride = 16
    threshold = 1
    nmsr = 30
    radius = 2
    detection = 5

    estimator = Estimator(stride,model_path,threshold,detection,nmsr,radius)

    cam_arg = "/Users/sedecim/Desktop/83511058-1-6.mp4"
    cam = cv2.VideoCapture(cam_arg)

    ret_val, image = cam.read()
    basic_shape = crop_image(image).shape

    
    frame_through_time = []
    # f_pos = 405;
    # for i in range(100):
    #     cam.set(cv2.CAP_PROP_POS_FRAMES,f_pos)
    #     ret_val, image = cam.read()
    #     frame_through_time.append(image)
    #     f_pos += 2

    # i = 0
    # for f in frame_through_time:
    #     cv2.imwrite("kun_" + str(i) + ".png",f)
    #     i+=1

    cam.set(cv2.CAP_PROP_POS_FRAMES,400)
    pose_through_time = []
    for i in range(100):
        # print(("frame:",i))
        # print("frames ready")
        ret,image = cam.read()
        image = crop_image(image)
        frame_through_time.append(image)
        (fac,fitted_img) = resize_img(image)
        poses = estimator.process_img(fitted_img, fac)
        
        if(len(poses) > 0):
            main_pose = max(poses)
            pose_through_time.append(main_pose)
            
        else:
            pose_through_time.append(None)
        
    print('frames ready')

    pose_filter = Pose_filter(Exp_filter,0.3);

    # drawn_image = np.zeros(basic_shape,'uint8')
    count = 0;
    while True:
        # drawn_image[:] = 0
        ind = count%len(pose_through_time)
        if pose_through_time[ind] != None:
            estimator.draw_pose_with_ease(pose_filter.push(pose_through_time[ind]),frame_through_time[ind])
            # estimator.draw_pose_with_ease((pose_through_time[ind]),frame_through_time[ind])
        
        cv2.imshow('cam_demo', frame_through_time[ind])
        
        time.sleep(0.04)

        count += 1;

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()




image = cam.read()
pose = estimator.read(image)