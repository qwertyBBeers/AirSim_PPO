# 이 부분에 PPO에 관한 action, state 등을 해 놓는다.
import numpy as np
import airsim
import time
import gym
import random
import typing

from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
 
class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address):
        super().__init__()

        self.state = {
            "lidar" : np.zeros((500, 2), dtype=np.dtype('f4')),
            "position" : np.zeros([1, 2]),
            "collision" : False,
            "position_state" : np.zeros([1, 2]),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)

        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        start_position = [[87, -4.63], [87, -3.70], [87, -2.06], [87, -0.29], [87, 1.43], [87, 4.78], [87, 6.56], [87, 8.4]]
        target_position = [[135.37, 25.26], [135.37, 23.26], [135.51, 26.87], [135.51, 28.69], [138.14, 28.26], [137.85, 26.71], [137.83, 23.28], [137.83, 23.26]]

        random_start = random.choice(start_position)

        start_x = random_start[0]
        start_y = random_start[1]
        start_z = 0

        start_index = start_position.index(random_start)

        #여기서 설정한 target_pos는 그냥 내가 쓰기 위해 설정하는 것일 뿐이다.
        self.target_pos = target_position[start_index]

        position = airsim.Vector3r(start_x, start_y, start_z)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision=True)

        self.drone.moveToPositionAsync(start_x, start_y, start_z, 3).join()
        self.drone.moveByVelocityAsync(1, 0.0, 1, 0).join()
        # 초기 드론 위치를 여기서 설정
    
    def _get_obs(self):

        self.drone_state = self.drone.getMultirotorState()

        #이 부분을 수정해야 함
        self.state["position"] = np.array([
            (self.target_pos[0]-self.drone_state.kinematics_estimated.position.x_val),
            (self.target_pos[1]-self.drone_state.kinematics_estimated.position.y_val)
            ])
        
        # 너무 멀리 갔을 때를 위해 설정        
        self.state["position_state"]= np.array([self.drone_state.kinematics_estimated.position.x_val,self.drone_state.kinematics_estimated.position.y_val])

        
        #충돌에 대한 정보 업데이트
        collision = self.drone.simGetCollisionInfo().has_collided
        
        #lidar 정보 업데이트    
        self.state["lidar"] = self.lidar_obs()
        self.state["collision"] = collision
        return self.state
    
    def lidar_obs(self):
        #총 500개의 배열에 lidar 정보가 들어온 만큼이 들어간다.
        lidar_data = self.drone.getLidarData()
        
        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        self.points_ = points[:,:-1]

        result_array = np.zeros((500, 2), dtype=np.dtype('f4'))
        result_array[:self.points_.shape[0]] = self.points_

        return result_array

    def _do_action(self, action):
        yaw_rate = action*30
        
        #x축 속도 고정, yaw의 회전만으로 장애물 회피
        self.drone.moveByVelocityZBodyFrameAsync(
            vx = 5,
            vy = 0.0,
            z = 1.5,
            duration = 3,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate= float(yaw_rate))
        )  

    # def make_batch(self, x):
    #     return np.expand_dims(x, axis=0)

    def AF(self):
        #distance 로 주는 방식
        att_gain = 0.003
        distance = np.linalg.norm([self.state["position"][0],self.state["position"][1]])
        # print(self.target_pos[0])
        # print(self.start_x)
        # max_dis = np.linalg.norm(self.target_pos[0]-self.start_pos[0], self.target_pos[1]-self.start_pos[1])
        max_dis = np.sqrt((self.target_pos[0]-self.start_x)**2+(self.target_pos[1]-self.start_y)**2)
        att_Force = att_gain*(max_dis - distance)
        # print(att_Force)

    
    def RF(self):
        rel_gain = 0.001
        rel_sum = 0
        obstacle_bound = 5
        rel_u = 0
        for obs_xy in self.points_:
            
            obs_dis = np.linalg.norm([obs_xy[0], obs_xy[1]])
            obs_dis = 0.1 if obs_dis <= 0.1 else obs_dis
            
            # print(obs_dis)
            if obs_dis < obstacle_bound:
                if obs_dis != 0:
                    rel_u= (1/(obs_dis) - 1/(obstacle_bound))**2                    
                else:
                    pass
                # print(rel_u)
            else:
                rel_u = 0
            rel_sum += rel_u
        rel_Force = 1/2*rel_gain*rel_sum
        return rel_Force

    def _compute_reward(self):
        #reward를 어떻게 주어줄 지에 대해서 작성
        goal = 0
        done = 0
        collision = False
        x_dis = self.target_pos[0] - self.state["position_state"][0]  
        y_dis = self.target_pos[1] - self.state["position_state"][1]
        
        if self.state['collision'] == True:
            done = 1
            collision = -10
            print("++++++++++++++++++++++++AF++++++++++++++++++++++++")
            print(self.AF())
            print("++++++++++++++++++++++++RF++++++++++++++++++++++++")
            print(self.RF())
            
        elif x_dis>=100 or x_dis<-100 or y_dis>=100 or y_dis<-100:
            done=1

        elif abs(self.state["position"][0])<5 and abs(self.state["position"][1])<5:
            done = 1
            goal = 100
            print("++++++++++++++++++++++++AF++++++++++++++++++++++++")
            print(self.AF())
            print("++++++++++++++++++++++++RF++++++++++++++++++++++++")
            print(self.RF())
            print("GOAL REACHED")

        else:
            done = 0
        
        APF = -self.AF() - self.RF()

        reward = APF + collision + goal
        
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()
