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
        self.yolo_model = YOLO('checkpoints/yolo11x.pt')  # 'yolov8n.pt'是YOLOv8 nano版本的预训练模型文件名，根据实际情况替换
        if os.path.exists('/home/admin/workspace/job/logs/'):
            self.handle = open('/home/admin/workspace/job/logs/user.log', 'w')
        else:
            self.handle = open('user.log', 'w')
        self.handle.write("model loaded\n")
        self.handle.flush()
        self.foreward = {
            'angular': 0, # [-30, 30]
            'velocity': 80, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.backward = {
            'angular': 0, # [-30, 30]
            'velocity': -50, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.turnleft = {
            'angular': -20, # [-30, 30]
            'velocity': 10, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.turnright = {
            'angular': 20, # [-30, 30]
            'velocity': 10, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.carry = {
            'angular': 0, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: keep, 1: up, 2: down},
            'interaction': 3,
        }
        self.drop = {
            'angular': 0, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: keep, 1: up, 2: down},
            'interaction': 4, # {0: stand, 1: jump, 2: crouch, 3: carry, 4: drop, 5: open door}
        }

        self.state_list = ['searching_person', 'approaching_person', 'searching_truck', 'approaching_truck', 'approaching_bench']
        self.pose = {
            'position': [0, 0, 0],
            'orientation': 0,
        }
        self.trcuk_list = ['truck', 'bus']
        self.bench_list = ['bench', 'skateboard', 'suitcase', 'chair', 'boat']
        self.reset(reference_text, reference_image)

    def reset(self, reference_text=None, reference_image=None):
        self.reference_text = reference_text
        self.reference_image = reference_image
        self.person_success = False
        self.state = 'searching_person'
        self.idx = 0
        print('searching_person')
        self.handle.write('searching_person\n')
        self.handle.flush()
        self.carry_flag = False
        self.search_counter = 0

    def predicts(self, ob, success):
        action = self.plan(ob, success)
        self.update_pose(action)
        if action['interaction'] == 3:
            self.carry_flag = True
        else:
            self.carry_flag = False
        return action
   
    def plan(self, ob, success):
        self.idx += 1
        if self.idx == 500:
            return self.drop
        self.handle.write('Step %d\n'%self.idx)
        self.handle.flush()

        ob = base64.b64decode(ob)
        ob = cv2.imdecode(np.frombuffer(ob, np.uint8), cv2.IMREAD_COLOR)
        
        results = self.yolo_model(source=ob,imgsz=640,conf=0.05) 
        boxes = results[0].boxes  # 获取所有检测框

        # Display image and YOLO detection results
        for box in boxes:
            res_ = box.xywh
            x0, y0, w_, h_ = res_[0].tolist()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(ob, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls = int(box.cls.item())
            label = self.yolo_model.names[cls]
            cv2.putText(ob, label+str(int(y0))+'_'+str(int(w_)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('YOLO Detection', ob)
        cv2.waitKey(1)
        
        if self.state == 'searching_person':
            for box in boxes:
                cls = int(box.cls.item())
                con = box.conf.item()
                if self.yolo_model.names[cls] == 'person' and con > 0.5:
                    self.handle.write("person detected\n")
                    print('person detected')
                    self.handle.flush()
                    self.state = 'approaching_person'
                    print('approaching_person')
                    self.search_counter = 0
                    return self.approach(box)
            self.search_counter += 1
            if self.search_counter < 20:
                return self.search_person()
            else: 
                return self.random_search()
                
        elif self.state == 'approaching_person':
            if self.carry_flag == True and success:
                self.handle.write("carry person success!!!!!!!\n")
                print('carry person success!!!!!!!')
                self.handle.flush()
                self.person_success = True
                self.state = 'searching_truck'
            else:
                for box in boxes:
                    cls = int(box.cls.item())
                    con = box.conf.item()
                    if self.yolo_model.names[cls] == 'person' and con > 0.5:
                        if self.approached(box, [420,230]):
                            self.handle.write("carry!!!!!!!!!!!!!!!!!!\n")
                            return self.carry
                        else:
                            return self.approach(box)
            return self.foreward

        elif self.state == 'searching_truck':
            for box in boxes:
                cls = int(box.cls.item())
                if self.yolo_model.names[cls] in self.trcuk_list and box.conf.item() > 0.3:
                    print('truck detected')
                    self.handle.write("truck detected\n")
                    self.handle.flush()
                    self.state = 'approaching_truck'
                    print('approaching_truck')
                    self.search_counter = 0
                    return self.approach(box)
            self.search_counter += 1
            if self.search_counter < 50:
                return self.search_truck()
            else:
                return self.random_search()
        
        elif self.state == 'approaching_truck':
            truck_box = None
            self.search_counter += 1
            for box in boxes:
                cls = int(box.cls.item())
                if self.yolo_model.names[cls] in self.bench_list and box.conf.item() > 0.1:
                    self.handle.write("approaching bench\n")
                    print('approaching bench')
                    self.state = 'approaching_bench'
                    self.search_counter = 0
                    return self.approach(box)
                if self.yolo_model.names[cls] in self.bench_list:
                    truck_box = box
            if truck_box is not None:
                return self.approach(truck_box)
            if self.search_counter < 20:
                return self.foreward
            else:
                return self.random_search()
            
        
        elif self.state == 'approaching_bench': 
            for box in boxes:
                cls = int(box.cls.item())
                if self.yolo_model.names[cls] in self.bench_list and box.conf.item() > 0.1:
                    self.handle.write("approaching bench\n")
                    print('approaching bench')
                    if self.approached(box, [400,270]):
                        self.handle.write("drop!!!!!!!!!!!!!!!!!!\n")
                        print('drop!!!!!!!!!!!!!!!!!!')
                        return self.drop
                    else:
                        return self.approach(box)
            return self.foreward
        return self.random_search()
    
    def random_search(self):
        random_action = {
            'angular': np.random.uniform(0, 30),
            'velocity': np.random.uniform(-10, 30),
            'viewport': 0,
            'interaction': 0
        }
        return random_action
                        

    def approach(self, box):
        res_ = box.xywh
        x, y, w_, h_ = res_[0].tolist()
        
        if x < 270:
            action = {'angular': np.clip((x-320)/5,-30,30), 'velocity': 0, 'viewport': 0, 'interaction': 0}
            return action
        elif x > 370:
            action = {'angular': np.clip((x-320)/5,-30,30), 'velocity': 0, 'viewport': 0, 'interaction': 0}
            return action
        else:
            return self.foreward
        
    def approached(self, box, threshold=[400, 230]):
        res_ = box.xywh
        x0, y0, w_, h_ = res_[0].tolist()
        if y0 > threshold[0] or w_>threshold[1]: 
            return True
        return False
        
    def search_person(self):
        if 'right' in self.reference_text[0]:
            return self.turnright
        elif 'left' in self.reference_text[0]:
            return self.turnleft
        else:
            return self.random_search()
        
    def search_truck(self):
        if self.pose['position'][0]**2 + self.pose['position'][1]**2 < 10000:
            return self.turnright
        else:
            action  = self.move([0,0])
            return action

    def move(self, target_position=[0, 0]):
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
    
    def update_pose(self, action):
        self.handle.write(json.dumps(action) + '\n')
        self.handle.flush()
        print(self.state)
        print(action)
        self.pose['position'][0] += action['velocity'] * np.sin(np.radians(self.pose['orientation']))
        self.pose['position'][1] += action['velocity'] * np.cos(np.radians(self.pose['orientation']))
        self.pose['orientation'] += action['angular']
        print(self.pose)


