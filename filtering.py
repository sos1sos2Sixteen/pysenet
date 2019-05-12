import random
import math
from estimator import *

# exponetial smoothing 
class Exp_filter():
    def __init__(self):
        self.last_output = None

    def reset(self):
        self.last_output = None
        
    def push(self, data, alpha):
        if self.last_output != None:
            self.last_output = alpha * data + (1 - alpha) * self.last_output
        else:
            self.last_output = data
        return self.last_output


def alpha(rate, cutoff):
    tau = 1.0 / (2 * math.pi * cutoff)
    te = 1.0 / rate
    return 1.0 / (1.0 + tau / te)

# one euro filter by Casiez et al
class Euro_filter():
    def __init__(self):
        self.min_cutoff = 0.01
        self.beta = 0.1
        self.dcutoff = 1
        
        self.xfilt = Exp_filter()
        self.dxfilt = Exp_filter()
        
        self.dx = 0
        self.first_time = True

    def reset(self):
        self.xfilt.reset()
        self.dxfilt.reset()
        self.dx = 0
        self.first_time = True
    
    def push(self, data, rate):
        if self.first_time:
            self.first_time = False
            self.dx = 0
        else:
            self.dx = (data - self.xfilt.last_output) * rate
        edx = self.dxfilt.push(self.dx, alpha(rate, self.dcutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        # cutoff = get_cutoff(edx)
        al = alpha(rate,cutoff)
        # print((rate,cutoff,al))
        return self.xfilt.push(data, alpha(rate, cutoff))

class Pose_filter():
    def __init__(self):
        self.fx_instance = []
        self.fy_instance = []

        for i in range(17):
            self.fx_instance.append(Euro_filter())
            self.fy_instance.append(Euro_filter())

    def reset(self):
        for ft in self.fx_instance:
            ft.reset()
        for ft in self.fy_instance:
            ft.reset()

    def set_beta(self, b):
        for ft in self.fx_instance:
            ft.beta = b
        for ft in self.fy_instance:
            ft.beta = b
    
    def set_min_cut(self, mc):
        for ft in self.fx_instance:
            ft.min_cutoff = mc
        for ft in self.fy_instance:
            ft.min_cutoff = mc
    
    def push(self, pose, rate):
        res = Pose()
        res.score = pose.score
        for i in range(17):
            key = pose.keypoints[i]
            if key != None:
                rp = (
                    int(self.fx_instance[i].push(key.pos[0], rate)),
                    int(self.fy_instance[i].push(key.pos[1], rate))
                )
                res.keypoints[i] = Keypoint(rp,key.key_id,key.score)
            else:
                # clear history
                # self.fx_instance[i].reset()
                # self.fy_instance[i].reset()
                pass
        return res

class Pose_filter_exp():
    def __init__(self):
        self.fx_instance = []
        self.fy_instance = []

        for i in range(17):
            self.fx_instance.append(Exp_filter())
            self.fy_instance.append(Exp_filter())

    def reset(self):
        for ft in self.fx_instance:
            ft.reset()
        for ft in self.fy_instance:
            ft.reset()

    def set_beta(self, b):
        for ft in self.fx_instance:
            ft.beta = b
        for ft in self.fy_instance:
            ft.beta = b
    
    def set_min_cut(self, mc):
        for ft in self.fx_instance:
            ft.min_cutoff = mc
        for ft in self.fy_instance:
            ft.min_cutoff = mc
    
    def push(self, pose, rate):
        res = Pose()
        res.score = pose.score
        for i in range(17):
            key = pose.keypoints[i]
            if key != None:
                rp = (
                    int(self.fx_instance[i].push(key.pos[0],0.5)),
                    int(self.fy_instance[i].push(key.pos[1],0.5))
                )
                res.keypoints[i] = Keypoint(rp,key.key_id,key.score)
            else:
                # clear history
                self.fx_instance[i].reset()
                self.fy_instance[i].reset()
        return res

