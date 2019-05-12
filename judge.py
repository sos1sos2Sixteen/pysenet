# 评分  输入pose为类
import math
from estimator import partNames
from body_region import regions
cal_vec_order = [
    (5, 0),  # 00 leftShoulder - nose
    (6, 0),  # 01 rightShoulder - nose
    (1, 0),  # 02 leftEye - nose
    (2, 0),  # 03 rightEye - nose
    (3, 1),  # 04 leftEar - leftEye
    (4, 2),  # 05 rightEar - rightEye
    (7, 5),  # 06 leftElbow - leftShoulder        left_up_arm
    (8, 6),  # 07 rightElbow - rightShoulder      right_up_arm
    (9, 7),  # 08 leftWrist - leftElbow           left_down_arm
    (10, 8),  # 09 rightWrist - rightElbow         right_down_arm
    (11, 5),  # 10 leftHip - leftShoulder          left_body
    (12, 6),  # 11 rightHip - rightShoulder        right_body
    (13, 11),  # 12 leftKnee - leftHip              left_up_leg
    (14, 12),  # 13 rightKnee - rightHip            right_up_leg
    (15, 13),  # 14 leftAnkle - leftKnee            left_down_leg
    (16, 14),  # 15 rightAnkle - rightKnee          right_down_leg

    (2, 1),  # 16 rightEye - leftEye              eyes
    (6, 5),  # 17 rightShoulder - leftShoulder    shoulders
    (12, 11),  # 18 rightHip - leftHip              hips
]

cal_angle_order = [
    (6, 8),  # 38 left_up_arm + left_down_arm       left_elbow
    (7, 9),     # right_up_arm + right_down_arm     right_elbow
    (6, 10),    # left_up_arm + left_body           left_shoulder
    (7, 11),    # right_up_arm + right_body         right_shoulder
    (12, 18),    # left_up_leg + hips           left_hip
    (13, 18),    # right_up_leg + hips         right_hip
    (12, 14),   # left_up_leg + left_down_leg       left_knee
    (13, 15),  # 45 right_up_leg + right_down_leg       right_knee
]

cal_angle_horiz = [
    16,
    17,
    18  # 48
]

vecNames = [
    'left_shoulder_nose_x',  # 0,
    'left_shoulder_nose_y',  # 1,
    'right_shoulder_nose_x',  # 2,
    'right_shoulder_nose_y',  # 3,
    'left_eye_nose_x',  # 4,
    'left_eye_nose_y',  # 5,
    'right_eye_nose_x',  # 6,
    'right_eye_nose_y',  # 7,
    'left_ear_eye_x',  # 8,
    'left_ear_eye_y',  # 9,
    'right_ear_eye_x',  # 10,
    'right_ear_eye_y',  # 11,
    'left_up_arm_x',  # 12,
    'left_up_arm_y',  # 13,
    'right_up_arm_x',  # 14,
    'right_up_arm_y',  # 15,
    'left_down_arm_x',  # 16,
    'left_down_arm_y',  # 17,
    'right_down_arm_x',  # 18,
    'right_down_arm_y',  # 19,
    'left_body_x',  # 20,
    'left_body_y',  # 21,
    'right_body_x',  # 22,
    'right_body_y',  # 23,
    'left_up_leg_x',  # 24,
    'left_up_leg_y',  # 25,
    'right_up_leg_x',  # 26,
    'right_up_leg_y',  # 27,
    'left_down_leg_x',  # 28,
    'left_down_leg_y',  # 29,
    'right_down_leg_x',  # 30,
    'right_down_leg_y',  # 31,
    'eyes_x',  # 32,
    'eyes_y',  # 33,
    'shoulders_x',  # 34,
    'shoulders_y',  # 35,
    'hips_x',  # 36,
    'hips_y',  # 37,
    'left_elbow_cos',  # 38,
    'right_elbow_cos',  # 39,
    'left_shoulder_cos',  # 40,
    'right_shoulder_cos',  # 41,
    'left_hip_cos',  # 42,
    'right_hip_cos',  # 43,
    'left_knee_cos',  # 44,
    'right_knee_cos',  # 45,

    'eyes_cos',  # 46,
    'shoulders_cos',  # 47,
    'hips_cos',  # 48,
]

# part_to_keypoint_def = {
#     'left_arm': ('leftShoulder', 'leftElbow', 'leftWrist'), 
#     'right_arm': ('rightShoulder', 'rightElbow', 'rightWrist'), 
#     'left_leg': ('leftHip', 'leftKnee', 'leftAnkle'),
#     'right_leg': ('rightHip', 'rightKnee', 'rightAnkle'), 
#     'head': ('nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar')
# }

part_to_keypoint_def = {
    'left_arm': ('leftElbow', 'leftWrist'), 
    'right_arm': ('rightElbow', 'rightWrist'), 
    'left_leg': ('leftKnee', 'leftAnkle'),
    'right_leg': ('rightKnee', 'rightAnkle'), 
    'head': ('nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar')
}

partname_to_keyid_dic = {}

for key in list(part_to_keypoint_def.keys()):
    tmp = ()
    for v in part_to_keypoint_def[key]:
        tmp += (partNames.index(v),)
    partname_to_keyid_dic[key] = tmp
# print('partname_to_keyid_dic:' + str(partname_to_keyid_dic))

part_to_vec_def = {
    'others':(),
    'head': ('eyes_cos',),
    'left_arm': ('left_elbow_cos', 'left_shoulder_cos'), 
    'right_arm': ('right_elbow_cos', 'right_shoulder_cos'), 
    'left_leg': ('left_hip_cos', 'left_knee_cos'),
    'right_leg': ('right_hip_cos', 'right_knee_cos'), 
    'shoulder': ('shoulders_cos',),
    'hip': ('hips_cos',)
}

partname_to_vecid_dic = {}
for key in list(part_to_vec_def.keys()):
    tmp = ()
    for v in part_to_vec_def[key]:
        tmp += (vecNames.index(v),)
    partname_to_vecid_dic[key] = tmp
# print('partname_to_vecid_dic:' + str(partname_to_vecid_dic))

def cal_vec(keypoint1, keypoint2):
    t = []

    if(keypoint1 == None or keypoint2 == None):
        #print("keypoint " + str(tp[0]) +  "or keypoint "+ str(tp[1]) + " lose in the pose!")
        # print("keypoint lose in the pose!")
        for x1 in range(2):
            t.append(0)
    else:
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

# 单位化
def unitization(vec):
    mode = math.sqrt(sum([x * x for x in vec]))
    t = []
    for x in vec:
        if(mode == 0):
            t.append(0)
        else:
            t.append(x / mode)
    return tuple(t)

# 计算余弦相似度
def cos_similarity(vec1, vec2):
    fenzi = sum([x * y for x, y in zip(vec1, vec2)])
    fenmu = math.sqrt(sum([x * x for x in vec1])) * \
        math.sqrt(sum([x * x for x in vec2]))
    return fenzi / fenmu

# 求特征向量
def vectorize(pose):
    vectors = []
    for tp in cal_vec_order:
        vec = cal_vec(pose.keypoints[tp[0]], pose.keypoints[tp[1]])
        vectors.append(vec)

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
        (x, y) = vectors[i]
        res.append(x)
        res.append(y)
    for i in range(len(cal_angle_order) + len(cal_angle_horiz)):
        k = i + len(cal_vec_order)
        res.append(vectors[k])

    return res

judge_std = {
    'others':[math.pi/36, math.pi*2/36],
    'head':[math.pi/36, math.pi*2/36],
    'left_arm':[math.pi/36, math.pi*2/36],
    'right_arm':[math.pi/36, math.pi*2/36],
    'left_leg':[math.pi/36, math.pi*2/36],
    'right_leg':[math.pi/36, math.pi*2/36],
    'shoulder':[math.pi/36, math.pi*2/36],
    'hip':[math.pi/36, math.pi*2/36],
}

# 判断
def qualify(target_vec, source_vec, judge_std):
    flag = True
    report = []  # 具体错误维数
    part_report = []  # 错误部分
    correctness = []    # 修正值
    i = 0
    for pname in list(judge_std.keys()):
        if pname == 'others':
            correctness.append(-1)
            continue
        vecids = partname_to_vecid_dic[pname]
        correct_sum = 0
        for k in vecids:
            if k >= 38: # 余弦值转弧度比较
                (left, right) = cal_range(source_vec[k], judge_std[pname][0])
                target_radians = math.acos(target_vec[k])
                if target_radians < left:
                    correct_sum += abs(target_radians - left)
                if target_radians > right:
                    correct_sum += abs(target_radians - right)

                print("k:" + str(k) + " value:" + str(target_radians) + " left:" + str(left) + " right:" + str(right))
                report.append((k,vecNames[k]))
                flag = False
        if correct_sum != 0:
            part_report.append(pname)

        correctness.append(correct_sum/len(regions[i]))
        print(correctness)
        print(len(regions[i]))
        i += 1
    correctness = normalization(correctness, judge_std)
    return (flag, correctness, [part_report, report])

def normalization(values, judge_std):
    i = 0
    for key in list(judge_std.keys()):
        if(values[i] == -1):
            i += 1
            continue
        if(values[i] > judge_std[key][1]):
            values[i] = 1
        else:
            values[i] = values[i] / judge_std[key][1]
        i += 1
    return values

# 身体部分转关键点id
def partname_to_keyid(part_names):
    keyid = set()
    for name in part_names:
        keyid.update(partname_to_keyid_dic[name])
    return keyid

# 计算误差范围
def cal_range(stdcos, offset_angle):
    std_radians = math.acos(stdcos)
    return (std_radians - offset_angle, std_radians + offset_angle)
