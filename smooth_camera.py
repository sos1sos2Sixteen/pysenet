import cv2
import numpy as np 
import time
from estimator import *
from filtering import *

def scale(x,bound,target):
    (a,b) = bound
    (ta,tb) = target
    s = (x-a)/(b-a)
    return s * (tb - ta) + ta

def transform_beta(inp):
    return scale(inp, (0,1000), (0,0.01))

def transform_fc(inp):
    return scale(inp, (0,1000), (0.000000001,0.01))


def change_beta(e):
    # beta value should be 0 or higher
    b = transform_beta(cv2.getTrackbarPos('beta','trace'))
    ef.set_beta(b)
    print(("beta = ", b))

def change_min_cutoff(e):
    # f_c_min should be 1 or lower
    mc = transform_fc(cv2.getTrackbarPos('min_cut','trace'))
    ef.set_min_cut(mc)
    print(("min_cutoff = ", mc))


cam = None
cam_arg = "/Users/sedecim/Desktop/83511058-1-6.mp4"
cam_arg = "/Users/sedecim/Desktop/hfhfhf.mp4"
cam_arg = 0
cam = cv2.VideoCapture(cam_arg)

model_path = "/Users/sedecim/Downloads/multi_person_mobilenet_v1_075_float.tflite"
stride = 16
threshold = 1
nmsr = 30
radius = 2
detection = 5
estimator = Estimator(stride,model_path,threshold,detection,nmsr,radius)

ef = Pose_filter()

def cv_init():
    cv2.namedWindow('trace')

    cv2.createTrackbar('beta','trace',0,1000,change_beta)
    cv2.createTrackbar('min_cut','trace',0,1000,change_min_cutoff)

    

cv_init()

# cam.set(cv2.CAP_PROP_POS_FRAMES,400)

# count = 0

start_t = 0
end_t = 0.041

ret,img = cam.read()
img= crop_image(img)
# drawn_image = np.zeros(img.shape,'uint8')
while True:
    start_t = time.perf_counter()
    ret, img = cam.read()
    img = crop_image(img)
    
    # drawn_image[:] = 0

    main_p = estimator.estimate(img)
    end_t = time.perf_counter()
    if(main_p != None):
        # validation
        estimator.draw_pose(ef.push(main_p,1.0/(end_t-start_t)), img,-100)

    cv2.imshow('trace',img)

    # if count > 400:
    #     count = 0
    #     cam.set(cv2.CAP_PROP_POS_FRAMES,400)

    # time.sleep(0.03)

    # count += 1
    

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cam.release()
        break
cv2.destroyAllWindows()
