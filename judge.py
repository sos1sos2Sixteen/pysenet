# 评分  输入pose为类
import math

cal_vec_order = [
    (5, 0),     #00 leftShoulder - nose 
    (6, 0),     #01 rightShoulder - nose
    (1, 0),     #02 leftEye - nose
    (2, 0),     #03 rightEye - nose
    (3, 1),     #04 leftEar - leftEye
    (4, 2),     #05 rightEar - rightEye
    (7, 5),     #06 leftElbow - leftShoulder        left_up_arm
    (8, 6),     #07 rightElbow - rightShoulder      right_up_arm
    (9, 7),     #08 leftWrist - leftElbow           left_down_arm
    (10, 8),    #09 rightWrist - rightElbow         right_down_arm
    (11, 5),    #10 leftHip - leftShoulder          left_body
    (12, 6),    #11 rightHip - rightShoulder        right_body
    (13, 11),   #12 leftKnee - leftHip              left_up_leg
    (14, 12),   #13 rightKnee - rightHip            right_up_leg
    (15, 13),   #14 leftAnkle - leftKnee            left_down_leg
    (16, 14),   #15 rightAnkle - rightKnee          right_down_leg

    (2, 1),     #16 rightEye - leftEye              eyes
    (6, 5),     #17 rightShoulder - leftShoulder    shoulders
    (12, 11),   #18 rightHip - leftHip              hips
]

cal_angle_order = [
    (6, 8),     # left_up_arm + left_down_arm       left_lbow 
    (7, 9),     # right_up_arm + right_down_arm     right_elbow
    (6, 10),    # left_up_arm + left_body           left_shoulder
    (7, 11),    # right_up_arm + right_body         right_shoulder
    (12, 18),    # left_up_leg + hips           left_hip
    (13, 18),    # right_up_leg + hips         right_hip
    (12, 14),   # left_up_leg + left_down_leg       left_knee
    (13, 15),   # right_up_leg + right_down_leg       right_knee
]

cal_angle_horiz = [
    16, 
    17,
    18
]
   
def cal_vec(keypoint1, keypoint2):
    t = []
   
    if(keypoint1 == None or keypoint2 == None):
        #print("keypoint " + str(tp[0]) +  "or keypoint "+ str(tp[1]) + " lose in the pose!")
        # print("keypoint lose in the pose!")
        for x1 in range(2):
            t.append(0)
    else:
        # print(str(keypoint1))
        pos1 = keypoint1.pos
        pos2 = keypoint2.pos
        for x1, x2 in zip(pos1, pos2):
            t.append(x1 - x2)
    return tuple(t)

def cal_cos(vec1, vec2):
    mode1 = math.sqrt(sum([x * x for x in vec1]))
    mode2 = math.sqrt(sum([x * x for x in vec2]))
    if(mode1 == 0 or mode2 == 0):
        return 1
    return sum([x * y for x, y in zip(vec1, vec2)]) / (mode1 * mode2)

def unitization(vec):
    mode = math.sqrt(sum([x * x for x in vec]))
    t = []
    for x in vec:
        if(mode == 0):
            t.append(0)
        else:
            t.append(x / mode)  
    return tuple(t)


def get_feature_vec(pose):
    for tp in cal_vec_order:
        vec = cal_vec(pose.keypoints[tp[0]], pose.keypoints[tp[1]])

def cos_similarity(vec1, vec2):
    fenzi = sum([x * y for x, y in zip(vec1, vec2)])
    fenmu = math.sqrt(sum([x * x for x in vec1])) * math.sqrt(sum([x * x for x in vec2]))
    return fenzi / fenmu

def vectorize(pose):
    vectors = []
    for tp in cal_vec_order:
        vec = cal_vec(pose.keypoints[tp[0]], pose.keypoints[tp[1]])
        vectors.append(vec)
        # (x,y) = vec
        # vectors.append(x)
        # vectors.append(y)

    for tp in cal_angle_order:
        ag = cal_cos(vectors[tp[0]], vectors[tp[1]])
        vectors.append(ag)

    for tp in cal_angle_horiz:
        ag = cal_cos(vectors[tp], (1, 0))
        vectors.append(ag)

    for i in range(len(cal_vec_order)):
        vectors[i] = unitization(vectors[i])

    res = []
    for i in range(len(cal_vec_order)):
        (x,y) = vectors[i]
        res.append(x)
        res.append(y)
    for i in range(len(cal_angle_order) + len(cal_angle_horiz)):
        k = i + len(cal_vec_order)
        res.append(vectors[k])

        
    return res
    
        
        
        

def judge(pose1, pose2):
    coss = []

    for tp in cal_vec_order:
        if(pose1.keypoints[tp[0]] == None or pose1.keypoints[tp[1]] == None):
            print("keypoint lose in the pose1!")
            vec1 = (0, 0)
        else:
            vec1 = cal_vec(pose1.keypoints[tp[0]].pos, pose1.keypoints[tp[1]].pos)
        if(pose2.keypoints[tp[0]] == None or pose2.keypoints[tp[1]] == None):
            print("keypoint lose in the pose2!")
            vec2 = (0, 0)
        else:
            vec2 = cal_vec(pose2.keypoints[tp[0]].pos, pose2.keypoints[tp[1]].pos)

        if(vec1[0] == 0 and vec1[1] == 0 or vec2[0] == 0 and vec2[1] == 0):
            coss.append(0)
        else:
            coss.append((cos_similarity(vec1, vec2) + 1) / 2)
    
    return coss

     

