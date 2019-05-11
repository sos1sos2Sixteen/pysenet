import os
import cv2
import numpy as np 
import tensorflow as tf
import queue
import random
import json


partNames = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle"
]


partIDs = {}
for i in range(len(partNames)):
    partIDs[partNames[i]] = i

poseChain = [
    ['nose', 'leftEye'], 
    ['leftEye', 'leftEar'], 
    ['nose', 'rightEye'],
    ['rightEye', 'rightEar'], 
    ['nose', 'leftShoulder'],
    ['leftShoulder', 'leftElbow'], 
    ['leftElbow', 'leftWrist'],
    ['leftShoulder', 'leftHip'],
    ['leftHip', 'leftKnee'],
    ['leftKnee', 'leftAnkle'], 
    ['nose', 'rightShoulder'],
    ['rightShoulder', 'rightElbow'], 
    ['rightElbow', 'rightWrist'],
    ['rightShoulder', 'rightHip'], 
    ['rightHip', 'rightKnee'],
    ['rightKnee', 'rightAnkle']
]

parentChildrenTuple = []
parentToChildEdges = []
childToParentEdges = []

for i in range(len(poseChain)):
    parentChildrenTuple.append([
        partIDs[poseChain[i][0]],
        partIDs[poseChain[i][1]]
    ])
    
    parentToChildEdges.append(partIDs[poseChain[i][1]])
    childToParentEdges.append(partIDs[poseChain[i][0]])                           
    

class Part():
    def __init__(self, _coord, _part_id):
        (x,y) = _coord
        self.coord = (x,y)
        self.part_id = _part_id

    def __str__(self):
        return "Part<" + str(self.part_id) + ">" + str(self.coord)

class PartWithScore():
    def __init__(self, _part, _score):
        self.part = _part
        self.score = _score
    
    def __str__(self):
        return "PwS(" + str(self.part) + "," + str(self.score) + ")"

    def __lt__(self, other):
        return self.score > other.score

class Keypoint():
    def __init__(self, _pos, _key_id, _score):
        (x,y) = _pos
        self.pos = (x,y)
        self.key_id = _key_id
        self.score = _score

    def __str__(self):
        return "Keypoint<" + str(self.key_id) + "," + str(self.score) + ">" + str(self.pos)

    def is_outside(self, size):
        (width, height) = size
        (x,y) = self.pos
        if x < 0 or x >= width or y < 0 or y >= height:
            return True
        else:
            return False

class Pose():
    def __init__(self):
        self.keypoints = []
        for i in range(17):
            self.keypoints.append(None)
        self.score = 0.0;

    def __str__(self):
        count = 0;
        for key in self.keypoints:
            if not key == None:
                count += 1
        return "Pose[" + str(self.score) + "](with " + str(count) + " parts)"


    def resize(self,factor):
        (fac_x, fac_y) = factor
        for key in self.keypoints:
            if key != None:
                (x,y) = key.pos
                key.pos = (
                    int(round(x * fac_x)),
                    int(round(y * fac_y))
                )

    def eliminate_false_positive(self, size):
        for i in range(len(self.keypoints)):
            if self.keypoints[i].is_outside(size):
                self.keypoints[i] = None

    def count_key(self):
        res = 0;
        for key in self.keypoints:
            if key != None:
                res += 1

        return res

    def serialize(self):
        dic = {
            'score' : self.score,
            'keypoints' : []
        }
        for key in self.keypoints:
            if not key == None:
                (x,y) = key.pos
                pos_dic = {
                    'x' : x,
                    'y' : y
                }
                key_dic = {
                    'id' : key.key_id,
                    'name' : partNames[key.key_id],
                    'score' : key.score,
                    'position' : pos_dic
                }
                dic['keypoints'].append(key_dic)
        return dic

    def __lt__(self, other):
            return self.score < other.score


def distance_squared(a,b):
    (xa,ya) = a
    (xb,yb) = b
    dist = ((xa - xb) ** 2) + ((ya - yb) ** 2)
    return dist

def within_nms_radius(key_id, nms_squared, poses, root_pos):
    for pose in poses:
        correspondingKey = pose.keypoints[key_id]
        if correspondingKey != None and distance_squared(correspondingKey.pos, root_pos) < nms_squared:
            return True
    return False

def score_is_local_maximum_in_window(part_id, scr, coord, radius, scores):
    local_maximum = True

    (x,y) = coord

    y_start = max(y - radius, 0)
    y_end   = min(y + radius + 1, 23)

    for y_current in range(y_start,y_end):
        x_start = max(x - radius,0)
        x_end   = min(x + radius + 1, 17)

        for x_current in range(x_start,x_end):
            # record.append((coord,y_current,x_current,scores[0,y_current,x_current,part_id],scr))
            if scores[0,y_current,x_current,part_id] > scr:
                local_maximum = False
                break
        if not local_maximum:
            break
    return local_maximum

def build_part_queue(threshold, radius, scores):
    part_queue = queue.PriorityQueue()

    for i in range(23):
        for j in range(17):
            for p in range(17):
                if(scores[0,i,j,p]) < threshold:
                    continue
                if score_is_local_maximum_in_window(p, scores[0,i,j,p], (j,i), radius, scores):
                    part_queue.put(PartWithScore(Part((j,i),p),scores[0,i,j,p]))

    return part_queue


def get_offset_vector(part_id, coord, offsets):
    (x,y) = coord
    y_tar = int(round(offsets[0,y,x,part_id]))
    x_tar = int(round(offsets[0,y,x,part_id + 17]))

    return (x_tar,y_tar)

def coord_to_pos(part_id, coord, offsets, stride):
    (x,y) = coord
    (x_off, y_off) = get_offset_vector(part_id, coord, offsets)
    return (x * stride + x_off, y * stride + y_off)

def clamp(a, mini, maxi):
    if a < mini:
        return mini
    if a > maxi:
        return maxi
    return round(a)

def pos_to_coord(pos, stride):
    (x,y) = pos
    return (
        clamp(int(round(x / stride)), 0, 17 - 1),
        clamp(int(round(y / stride)), 0, 23 - 1)
    )

def get_score(part_id, coord, scores):
    (x,y) = coord
    return scores[0,y,x,part_id]

def get_displace(edge_id, coord, is_backward, displace):
    
    channel = 0
    if is_backward:
        channel = 32
    (x,y) = coord
    # print((y,x,channel + edge_id,channel+edge_id+16))
    return(
        displace[0,y,x,channel + edge_id + 16],
        displace[0,y,x,channel + edge_id]
    )

def add_vec2(a,b):
    (xa, ya) = a
    (xb, yb) = b
    return (xa + xb, ya + yb)



class Estimator():

    def __init__(self, _stride, _model_path, _threshold, _max_detection, _nmsr, lmr):
        self.interpreter = tf.lite.Interpreter(model_path = _model_path)
        self.interpreter.allocate_tensors()

        self.input_detail = self.interpreter.get_input_details()
        self.output_detail = self.interpreter.get_output_details()

        self.o_y = 23
        self.o_x = 17
        self.stride = _stride
        self.threshold = _threshold
        self.max_detection = _max_detection
        self.nmsr = _nmsr
        self.local_maximum_radius = lmr

    def feed_net(self, croped_img):
        self.interpreter.set_tensor(self.input_detail[0]['index'],croped_img)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_detail[0]['index'])
        offsets = self.interpreter.get_tensor(self.output_detail[1]['index'])
        displace = self.interpreter.get_tensor(self.output_detail[2]['index'])

        return (scores,offsets,displace)

    def traverse_to_target_keypoint(
            self, 
            edge_id, 
            source_keypoint, 
            target_keypoint_id, 
            is_backward, 
            displace, 
            offsets,
            scores
        ):

        source_keypoint_indecies = pos_to_coord(source_keypoint.pos, self.stride)

        displacement = get_displace(edge_id, source_keypoint_indecies, is_backward, displace)

        displacedPoint = add_vec2(source_keypoint.pos, displacement)

        displacedPoint_indecies = pos_to_coord(displacedPoint, self.stride)

        (x,y) = displacedPoint_indecies

        offset_vector = get_offset_vector(target_keypoint_id, displacedPoint_indecies, offsets)

        score = get_score(target_keypoint_id, displacedPoint_indecies, scores)

        target_keypoint = add_vec2(displacedPoint, offset_vector)

        (x,y) = target_keypoint;

        return Keypoint((int(round(x)),int(round(y))), target_keypoint_id, score)

    def decode_pose(
        self,
        root,
        offsets,
        displace,
        scores
        ):
        instance = Pose()

        root_part = root.part
        root_score = root.score

        root_point = coord_to_pos(root_part.part_id, root_part.coord, offsets, self.stride)
        (x,y) = root_point
        instance.keypoints[root_part.part_id] = Keypoint(
            (int(round(x)), int(round(y))),
            root_part.part_id,
            root_score
        )

        edge = 16 - 1

        while edge >= 0:
            source_keypoint_id = parentToChildEdges[edge]
            target_keypoint_id = childToParentEdges[edge]

            if((instance.keypoints[source_keypoint_id] != None) and (instance.keypoints[target_keypoint_id] == None)):
                instance.keypoints[target_keypoint_id] = self.traverse_to_target_keypoint(
                    edge,
                    instance.keypoints[source_keypoint_id],
                    target_keypoint_id,
                    True,
                    displace,
                    offsets,
                    scores
                )
            edge -= 1;
        
        edge = 0
        while edge < 16:
            source_keypoint_id = childToParentEdges[edge]
            target_keypoint_id = parentToChildEdges[edge]

            if((instance.keypoints[source_keypoint_id] != None) and (instance.keypoints[target_keypoint_id] == None)):
                instance.keypoints[target_keypoint_id] = self.traverse_to_target_keypoint(
                    edge,
                    instance.keypoints[source_keypoint_id],
                    target_keypoint_id,
                    False,
                    displace,
                    offsets,
                    scores
                )
            edge += 1;

        return instance

    def decode_multiple_poses(self,scores, offsets, displace):
        squared_nmsr = self.nmsr ** 2
        part_queue = build_part_queue(self.threshold, self.local_maximum_radius, scores)
        poses = []

        while(len(poses) < self.max_detection) and (not part_queue.empty()):
            root = part_queue.get()
            root_pos = coord_to_pos(root.part.part_id, root.part.coord, offsets, self.stride)
            if(within_nms_radius(root.part.part_id, squared_nmsr, poses, root_pos)):
                continue
            pose = self.decode_pose(root,offsets,displace,scores)
            pose.eliminate_false_positive((257,353))
            pose.score = self.get_instance_score(poses, squared_nmsr, pose)
            poses.append(pose)

            
        
        return poses

    def get_instance_score(self, poses, snmsr, p):
        res = 0
        for key in p.keypoints:
            if key != None and not within_nms_radius(key.key_id, snmsr, poses, key.pos):
                res += key.score

        count = p.count_key()
        if count != 0:
            return res / count
        else:
            return -233

    def draw_pose(self, p, base_img, threshold):
        if p.score < threshold:
            return
        # random_color = (random.randint(50,250),random.randint(50,250),random.randint(50,250))
        random_color = (0,244,289);
        for key in p.keypoints:
            if(key != None):
                cv2.circle(base_img, key.pos, 3, random_color, 2)
        for edge in parentChildrenTuple:
            source = p.keypoints[edge[0]]
            target = p.keypoints[edge[1]]
            if source != None and target != None:
                cv2.line(base_img, source.pos, target.pos, random_color,1)
        
        if p.keypoints[5] != None:
            cv2.circle(base_img, p.keypoints[5].pos,5, (250,0,0),3)
    
    def draw_pose_with_ease(self, p, base_img):
        # random_color = (random.randint(50,250),random.randint(50,250),random.randint(50,250))
        random_color = (0,244,289);
        no_draw = [1,2,3,4]
        for key in p.keypoints:
            if(key != None):
                if(key.key_id in no_draw):
                    continue
                cv2.circle(base_img, key.pos, 3, random_color, 2)
        for edge in parentChildrenTuple:
            source = p.keypoints[edge[0]]
            target = p.keypoints[edge[1]]
            if source != None and target != None:
                if source.key_id in no_draw or target.key_id in no_draw:
                    continue
                cv2.line(base_img, source.pos, target.pos, random_color,1)
        if p.keypoints[5] != None and p.keypoints[6] != None:
            cv2.line(base_img, p.keypoints[5].pos, p.keypoints[6].pos, random_color,1)
        if p.keypoints[12] != None and p.keypoints[11] != None:
            cv2.line(base_img, p.keypoints[12].pos, p.keypoints[11].pos, random_color,1)
        
        if p.keypoints[5] != None:
            cv2.circle(base_img, p.keypoints[5].pos,5, (250,0,0),3)
    
    def draw_poses(self,poses,base_img, threshold):

        for p in poses:
            self.draw_pose(p, base_img, threshold)

    def process_img(self, img, factor):
        (scores,offsets,displace) = self.feed_net(img)
        poses = self.decode_multiple_poses(scores,offsets,displace)
        
        for p in poses:
            p.resize(factor)

        return poses
    
    def estimate(self, img):
        (fac, fitted_img) = resize_img(img)
        poses = self.process_img(fitted_img, fac)

        main_pose = None
        if(len(poses) > 0):
            main_pose = max(poses)

        return main_pose



def crop_image(img):
    # 8:11
    (y,x,channel) = img.shape
    # x_prime = (8/11.0)*y
    x_prime = y
    img = img[0:y, int((x-x_prime)/2):int((x+x_prime)/2)]
    return img

def resize_img(img):
    # 257,353
    fitted_img = cv2.resize(img, (257,353))
    (y,x,channel) = img.shape
    fitted_img = np.expand_dims(fitted_img, axis = 0)
    fitted_img = fitted_img.astype('float32')
    fitted_img = fitted_img / (255/2.0) - 1

    return ((x/257,y/353),fitted_img)