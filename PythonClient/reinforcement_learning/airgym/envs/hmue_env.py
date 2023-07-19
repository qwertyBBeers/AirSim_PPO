# 이 부분에 PPO에 관한 action, state 등을 해 놓는다.

import numpy as np
import airsim
import time
import gym
import random
from gym import spaces

import typing 

class AirSimEnv(AirSimEnv):
    def __init__(self, ip_address):
        super().__init__()

        self.state = {
            "lidar" : np.zeros((500, 2), dtype=np.dtype('f4')),
            "position" : np.zeros([1, 2]),
            "collision" : False,
            "goal_state" : np.zeros([1, 2]),
            "velocity" : np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.start= time.time()
        self._setup_flight()


    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        start_position = [[87, -7.02], [87, -4.63], [87, -3.70], [87, -2.06], [87, -0.29], [87, 1.43], [87, 4.78], [87, 6.56], [87, 8.4]]
        target_position = [[[135.37, 25.26], [135.37, 23.26], [135.51, 26.87], [135.51, 28.69], [138.14, 28.26], [137.85, 26.71], [137.83, 23.28], [137.83, 23.26]]]

        random_start = random.choice(start_position)

        start_x = random_start[0]
        start_y = random_start[1]
        start_z = 1.5

        start_index = start_position.index(random_start)

        #여기서 설정한 target_pos는 그냥 내가 쓰기 위해 설정하는 것일 뿐이다.
        self.goal_pose = random.choice(target_position[start_index])

        position = airsim.Vector3r(start_x, start_y, start_z)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision=True)
        self.time = 0

        self.drone.moveToPositionAsync(start_x, start_y, start_z, 3).join()
        self.drone.moveByVelocityAsync(1, 0.0, -0.8, 1.5).join()
        # 초기 드론 위치를 여기서 설정

    def lidar_obs(self):
        #총 500개의 배열에 lidar 정보가 들어온 만큼이 들어간다.
        lidar_data = self.drone.getLidarData().point_cloud
        
        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        points_ = points[:,:-1]

        result_array = np.zeros((500, 2), dtype=np.dtype('f4'))
        result_array[:points_.shape[0]] = points_

        return result_array

    def _get_obs(self):

        self.drone_state = self.drone.getMultirotorState()

        #속도에 대한 입력값을 넣어주기 위함
        vx = self.drone_state.kinematics_estimated.linear_velocity.x_val
        vy = self.drone_state.kinematics_estimated.linear_velocity.y_val
        yaw_rate = self.drone_state.kinematics_estimated.angular_velocity.z_val

        #속도값 정보 업데이트
        self.state["velocity"] = self.make_batch(np.array([vx, vy, yaw_rate]))

        #이 부분을 수정해야 함
        self.state['position'] = np.array([
            (self.target_pos[0]-self.drone_state.kinematics_estimated.position.x_val),
            (self.target_pos[1]-self.drone_state.kinematics_estimated.position.y_val)
            ])
        
        #충돌에 대한 정보 업데이트
        collision = self.drone.simGetCollisionInfo().has_collided
        
        #lidar 정보 업데이트    
        self.state["lidar"] = self.lidar_obs()
        self.state["collision"] = collision
        raise self.state

    def make_batch(self, x):
        return np.expand_dims(x, axis=0)
    
    def _do_action(self, action):
        yaw_rate = action
        
        #x축 속도 고정, yaw의 회전만으로 장애물 회피
        self.drone.moveByVelocityZBodyFrameAsync(
            vx = 1,
            vy = 0.0,
            z = 1.5,
            duration = 3,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate= float(yaw_rate))
        )  


    def _compute_reward(self):
        #reward를 어떻게 주어줄 지에 대해서 작성
        raise reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()
