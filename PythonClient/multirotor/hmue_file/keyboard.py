import airsim
import os

import time
import sys

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

z = 0
duration = 1
vx = 0
vy = 0

def get_key_input():
    global vx, vy
    while True:
        key = input("input Key : ")
        if key.lower() == 'w':
            vx = 5
            vy = 0
            return vx, vy
        elif key.lower() == 'a':
            vx = 0
            vy = -5
            return vx, vy
        elif key.lower() == 's':
            vx = -5
            vy = 0
            return vx, vy
        elif key.lower() == 'd':
            vx = 0
            vy = 5
            return vx, vy
        else:
            vx = 0
            vy = 0
            return vx, vy

if __name__ == "__main__":
    while True:
        vx, vy = get_key_input()
        client.moveByVelocityZAsync(vx,vy,z,duration)
        print(vx)
        print(vy)
        print("------------------")
    