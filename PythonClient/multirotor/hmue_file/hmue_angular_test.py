import airsim
import os

import time
import sys

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

z = 1.5
duration = 1
vx = 0
vy = 0
yaw_rate = 0
def get_key_input():
    global vx, vy, yaw_rate
    while True:
        key = input("input Key : ")
        if key.lower() == 'w':
            vx = 5
            vy = 0
            yaw_rate = 0
            return vx, vy
        elif key.lower() == 'a':
            vx = 0
            vy = -5
            yaw_rate = -20
            return vx, vy
        elif key.lower() == 's':
            vx = -5
            vy = 0
            yaw_rate = -30
            return vx, vy
        elif key.lower() == 'd':
            vx = 0
            vy = 5
            yaw_rate = 20
            return vx, vy
        else:
            vx = 0
            vy = 0
            return vx, vy

if __name__ == "__main__":
    while True:
        vx, vy = get_key_input()

        # client.moveByVelocityAsync(5, 0, 0, duration=5).join()
        # client.moveByVelocityZBodyFrameAsync(vx = 5.0, vy = 0.0, z = 1.0, duration = 5).join()
        # client.rotateByYawRateAsync(yaw_rate=yaw_rate, duration=5).join()
        client.moveByVelocityZBodyFrameAsync(
            vx = 5,
            vy = 0.0,
            z = -5.0,
            duration = 3,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate= float(yaw_rate))
        )
        
        print(vx)
        print(vy)
        print("------------------")
    