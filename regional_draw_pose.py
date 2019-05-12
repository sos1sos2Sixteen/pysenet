import numpy as np 
from body_region import *
from estimator import Pose, Keypoint
import cv2

def hsl_to_rgb(hsl):
    (h,s,l) =  hsl
    if s == 0:
        L = int(255 * l)
        return (L,L,L)
    else:
        tmp2 = 0
        if l < 0.5:
            tmp2 = l * (1 + s)
        else:
            tmp2 = l + s - l * s
        tmp1 = 2 * l - tmp2

        tpr = h + 1/3
        tpg = h
        tpb = h - 1/3
        
        if tpr < 0:
            tpr += 1
        if tpr > 1:
            tpr -= 1

        r = 0
        if 6 * tpr < 1:
            r = tmp1 + (tmp2 - tmp1) * 6 * tpr
        elif 2 * tpr < 1:
            r = tmp2
        elif 3 * tpr < 2:
            r = tmp1 + (tmp2 - tmp1) * ((2/3) - tpr) * 6
        else:
            r = tmp1

        if tpg < 0:
            tpg += 1
        if tpg > 1:
            tpg -= 1

        g = 0
        if 6 * tpg < 1:
            g = tmp1 + (tmp2 - tmp1) * 6 * tpg
        elif 2 * tpg < 1:
            g = tmp2
        elif 3 * tpg < 2:
            g = tmp1 + (tmp2 - tmp1) * ((2/3) - tpg) * 6
        else:
            g = tmp1

        if tpb < 0:
            tpb += 1
        if tpb > 1:
            tpb -= 1

        b = 0
        if 6 * tpb < 1:
            b = tmp1 + (tmp2 - tmp1) * 6 * tpb
        elif 2 * tpb < 1:
            b = tmp2
        elif 3 * tpb < 2:
            b = tmp1 + (tmp2 - tmp1) * ((2/3) - tpb) * 6
        else:
            b = tmp1

        r = int(255 * r)
        g = int(255 * g)
        b = int(255 * b)

        return (r,g,b)
        

def draw_point(img, key, color):
    if(key != None):
        cv2.circle(img, key.pos, 3, color, 2)

def draw_line(img, genos, talos, color):
    if(genos != None and talos != None):
        cv2.line(img, genos.pos, talos.pos, color, 2)

def draw_region(img, pose, region, color):
    for ft in region:
        if ft[0]:
            # feature is line
            (genos, talos) = ft[1]
            draw_line(
                img,
                pose.keypoints[genos],
                pose.keypoints[talos],
                color)
        else:
            # feature is point
            draw_point(
                img,
                pose.keypoints[ft[1]],
                color
            )

def get_regional_color(judgement):
    if judgement == -1:
        return hsl_to_rgb((0.45, 0.65, 0.6))
    if judgement < 0.00000001:
        return (0,255,0)
    return hsl_to_rgb(((1 - judgement) * 0.25, 0.9, 0.5))

def regional_draw_pose(img, pose, judgements):
    if len(judgements) != len(regions):
        print("regional_draw_pose: invalid judgements data, giving up!")
        return
    
    for i in range(len(regions)):
        reg = regions[i]
        judgement = judgements[i]
        color = get_regional_color(judgement)
        draw_region(img, pose, reg, color)
