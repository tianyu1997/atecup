import time
import sys
import json
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from ultralytics import YOLO
# 加载预训练的YOLO模型，如果下载了特定版本的权重，请指定路径

class AlgSolution:

    def __init__(self, reference_text=None, reference_image=None):
        self.load_model()
        if os.path.exists('/home/admin/workspace/job/logs/'):
            self.handle = open('/home/admin/workspace/job/logs/user.log', 'w')
        else:
            self.handle = open('user.log', 'w')
        self.reset(reference_text, reference_image)
        self.map = Map()
        
    def load_model(self):
        self.yolo_model = YOLO('yolov11n.pt')

    

    def reset(self, reference_text=None, reference_image=None):
        self.reset_models()
        self.map.reset()
        self.init_map_falg = False
        self.reference_text = reference_text
        self.reference_image = reference_image
        self.idx = 0
        self.carry_flag = False
        self.get_target_list(self.reference_text, self.reference_image)
        self.target_id = 0
        self.target_name = self.target_list[self.target_id]
        self.target = None
    
    def reset_models(self):
        ## 重置有时序依赖的模型
        pass

    def get_target_list(self, reference_text)->list[str]:
        ## 从reference_text中提取目标列表， 目标必须包含“person”，最后一个目标为"stretcher"
        pass

    def init_Map(self, ob):
        ## init map 比如原地转一圈，记录周围的物体, 返回action,结束后设置init_map_flag = True
        pass

    def predicts(self, ob, success):
        self.map.update(ob, self.last_action)
        if not self.init_Map_falg:
            action = self.init_Map(ob)
        else:
            action = self.plan(ob, success)
        if action['interaction'] == 3:
            self.carry_flag = True
        else:
            self.carry_flag = False
        self.last_action = action
        return action
   
    def plan(self, ob, success):
        if self.target is None:
            action = self.search(ob)
        elif self.reached(ob):
            if self.target_name == 'person':
                action =  {'angular': 0, 'velocity': 0, 'viewport': 0, 'interaction': 3}
            elif self.target_name == 'stretcher':
                action =  {'angular': 0, 'velocity': 0, 'viewport': 0, 'interaction': 4}
            else:
                self.target_id += 1
                self.target_name = self.target_list[self.target_id]
                self.target = None
                action = self.search(ob)
        else:
            action = self.approach(ob)
        return action


    def search(self, ob):
        pass
                        

    def approach(self, ob):
        pass
        
        
    def reached(self, ob):
        pass
        

    def goto(self, target_position=[0, 0]):
        dx = target_position[0] - self.pose['position'][0]
        dy = target_position[1] - self.pose['position'][1]
        angle = np.arctan2(dx, dy)  # y-axis as 0 degrees, clockwise as positive
        angle_diff = angle - np.radians(self.pose['orientation'])
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        angle_diff = np.degrees(angle_diff)
        if abs(angle_diff) < 5:
            return {'angular': 0, 'velocity': 100, 'viewport': 0, 'interaction': 0}
        if angle_diff > 0:
            return {'angular': min(30, angle_diff), # [-30, 30]
            'velocity': 10, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0}
        elif angle_diff < 0:
            return {'angular': max(-30, angle_diff), # [-30, 30]
            'velocity': 10, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0}
    
    


class Pose():    
    def __init__(self, position=[0, 0], orientation=0):
        self.position = position
        self.orientation = orientation

class Object():
    def __init__(self, name, pose:Pose, size=[1,1]):
        self.name = name
        self.pose = pose
        self.size = size


class Map():
    def __init__(self, size=[1000, 1000]):
        self.reset(size)
    
    def reset(self, size=[1000, 1000]):
        self.size = size
        self.map = np.zeros(size)
        self.pose = Pose([0,0], 0)
        self.objects = dict()
    
    def update(self, obs):
        pass

    def add_object(self, obj):
        name = obj.name
        if name not in self.objects:
            self.objects[name] = dict({'item': [obj], 'map_id': len(self.objects), 'explored': False})
        else:
            self.objects[name]['item'].append(obj)
        self.map_update(obj)

    def map_update(self, obj):
        pass

    def render(self):
        pass

    def update_pose(self, action):
        self.pose['position'][0] += action['velocity'] * np.sin(np.radians(self.pose['orientation']))
        self.pose['position'][1] += action['velocity'] * np.cos(np.radians(self.pose['orientation']))
        self.pose['orientation'] += action['angular']
        print(self.pose)