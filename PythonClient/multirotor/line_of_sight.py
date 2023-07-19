import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# drone의 홈 위치를 가져옴
home = client.getHomeGeoPoint()
print("home:\n%s" % home)

target = home
target.latitude -= 1

# 드론과 특정 지점 간의 시야 테스트를 수행
result = client.simTestLineOfSightToPoint(target)
print("test line of sight from vehicle to\n%s\n\t:%s" %(target, result))

# 두 지점 사이의 시야 테스트를 수행
result = client.simTestLineOfSightBetweenPoints(home, target)
print("test line of sight from home to\n%s\n\t:%s" %(target, result))

# 월드의 범위를 가져옴
result = client.simGetWorldExtents()
print("world extents:\n%s\n\t-\n%s" %(result[0], result[1]))

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
