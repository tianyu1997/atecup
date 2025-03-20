import time
import sys
import json
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from ultralytics import YOLO

class AlgSolution:
    def __init__(self, reference_text=None, reference_image=None):
        self.load_model()
        log_path = '/home/admin/workspace/job/logs/user.log' if os.path.exists('/home/admin/workspace/job/logs/') else 'user.log'
        self.handle = open(log_path, 'w')
        
        self.map = Map()
        self.searcher = Searcher(self.map)
        self.planner = Planner(self.map)  
        self.target_manager = TargetManager(reference_text)
        self.reset(reference_text, reference_image)
        
    def reset(self, reference_text=None, reference_image=None):
        self.reset_models()
        self.map.reset()
        self.init_map_falg = False
        self.reference_text = reference_text
        self.reference_image = reference_image
        self.idx = 0
        self.carry_flag = False
        self.searcher.reset(self.map)
        self.planner.reset(self.map)
        self.target_manager.reset(reference_text)
        
    def load_model(self):
        self.yolo_model = YOLO('yolov11n.pt')
    
    def reset_models(self):
        # Reset models with temporal dependencies
        pass


    def init_Map(self, ob):
        ## init map 比如原地转一圈，记录周围的物体, 返回action,结束后设置init_map_flag = True
        pass

    def predicts(self, ob, success):
        self.map.update(ob, self.last_action)
        if not self.init_map_flag:
            action = self.init_map(ob)
        else:
            action = self.plan(ob, success)
        self.carry_flag = action['interaction'] == 3
        self.last_action = action
        return action
   
    def plan(self, ob, success):
        if self.target_manager.target_object is None:
            action, flag = self.search(ob)
            if flag: ## 确定找不到目标
                self.target_manager.pop()
        elif self.reached(ob):
            if self.target_name == 'person':
                if self.carry_flag and success: ## 抬起人成功切换下个目标
                    self.target_manager.carry_person=True
                    self.target_manager.next()
                    action = self.search(ob)
                else:
                    action =  self.get_action(0,0,0,3) ## 抬起人
            elif self.target_name == 'stretcher':
                action =  self.get_action(0,0,0,4) ## 放下人
            else: ## 切换下个目标
                self.target_manager.next()
                action = self.search(ob)
        else:
            action = self.approach(ob)
        return action

    def get_action(self, ang, vel, view, interaction):
        return {'angular': ang, 'velocity': vel, 'viewport': view, 'interaction': interaction}

    def search(self, ob):
        ## 搜索目标，返回action, flag, flag为True表示当前局部范围内找不到目标
        pass
                        
    def approach(self, ob):
        pass
        
    def reached(self, ob):
        pass


class Pose():    
    def __init__(self, position=[0, 0], orientation=0):
        self.position = position
        self.orientation = orientation

class Object():
    def __init__(self, name, pose:Pose, size=[1,1]):
        self.name = name
        self.pose = pose
        self.size = size
        self.expolre = False


class Map():
    ## SLAM类 
    # size: map的pixel尺寸
    # scale: 每个pixel的大小与观察者位移的比例
    # 每一步会根据上一步的action和当前的obs维护以下内容：
    #     object_map: 二维np.array, 二维地图，地图上的数字表明对应的object类型
    #     objects: 字典类型，记录看到过的物体，物体名字作为key，‘item’ 是Object类的list, mask指在object_map上对应的区域数字
    #     occupied_map:二维np.array,记录某个pixel位置是否被占用，也就是人能正常通过，不会碰撞，0是不会碰撞，1是会碰撞，门可能需要特殊处理
    #     explored_map:二维np.array,记录某个pixel位置是否被探索观察过
    #     observe_pose:Pose类型，记录当前观察者的位置（最好能结合obs考虑卡模型情况）
    #     in_house_flag: bool, 是否在房内
    #     in_room_flag: bool, 是否在房内的房间内
    def __init__(self, size=[1000, 1000], scale=100):
        self.size = size
        self.scale = scale
        self.reset(size, scale)
    
    def reset(self, size=None, scale=None):
        if size is not None:
            self.size = size
        if scale is not None:
            self.scale = scale
            
        self.objects = dict()
        self.object_map = np.zeros(size)
        
        self.occupied_map = np.zeros(size)
        self.explored_map = np.zeros(size)
        self.observe_pose = Pose([0,0], 0)
        self.in_house_flag = False
        self.in_room_flag = False
        
    def update(self, obs, action):
        pass

    def render(self):
        ## 返回3张图片格式的map，最好能标注出观测者的初始和当前pose
        pass

    # def add_object(self, obj: Object):
    #     name = obj.name
    #     if name not in self.objects:
    #         self.objects[name] = dict({'item': [obj], 'mask': len(self.objects)})
    #     else:
    #         self.objects[name]['item'].append(obj)
    #     self.map_update(obj)

    # def map_update(self, obj):
    #     pass

    def update_observe_pose(self, action):
        self.observe_pose['position'][0] += action['velocity'] * np.sin(np.radians(self.observe_pose['orientation']))
        self.observe_pose['position'][1] += action['velocity'] * np.cos(np.radians(self.observe_pose['orientation']))
        self.observe_pose['orientation'] += action['angular']
        print(self.observe_pose)

class TargetManager():
    ## 管理target的类
    #  根据ref信息生成target_list, 第一个目标是"stretcher"，最后一个目标为“person”
    #  根据任务进程切换目标
    def __init__(self, ref_text):
        self.reset(ref_text)

    def reset(self, ref_text):
        self.get_target_list(ref_text)
        self.target_id = 0
        self.target_name = self.target_list[self.target_id]
        self.target_history = []
        self.target_object = None
        self.carry_person=False

    def generate_target_list(self)->list:
        pass 

    def next(self):
        if not self.carry_person:
            self.target_history.append(self.target_object)
            self.target_id += 1
            self.target_name = self.target_list[self.target_id]
            self.target_object = None
        else:
            self.target_id -= 1
            self.target_object = self.target_history[self.target_id]

    def pop(self):
        self.target_id += -1
        self.target_name = self.target_list[self.target_id]
        self.target_object = self.target_history.pop()

class Searcher():
    ## 利用大模型根据ref信息和Map信息，以及当前的target_name寻找target_object
    ## 一种方案可选方案是输出value_map，根据value_map决定搜索优先级，且判断观测的到同名object是不是target_object
    #  需要判断当前局部空间是否探索完全（如一个房间内找遍，没找到人）
    
     def __init__(self, map:Map, ref_txt, ref_img):
         pass

     def reset():
         pass
         
     def load_model(self):
         pass 
         
     def search(self, target_name):
         pass
         

class Planner():
     ## 规划无碰撞路线，走到目标位置
     def __init__(self, map:Map):
         pass
         
     def reset(self):
         pass
         
     def goto(self, target_pose:Pose):
         pass
         
     def move(self, target_pose:Pose):
        dx = target_pose.position[0] - self.pose['position'][0]
        dy = target_pose.position[1] - self.pose['position'][1]
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
