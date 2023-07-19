import setup_path
import airsim

import sys
import time

print("""This script is designed to fly on the streets of the Neighborhood environment
and assumes the unreal position of the drone is [160, -1500, 120].""")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

print("arming the drone...")
client.armDisarm(True)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    client.hoverAsync().join()

time.sleep(1)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("take off failed...")
    sys.exit(1)

# AirSim uses NED coordinates so negative axis is up.
# z of -5 is 5 meters above the original launch point.
z = -25
print("make sure we are hovering at {} meters...".format(-z))
client.moveToZAsync(z, 1).join()

# see https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo

# this method is async and we are not waiting for the result since we are passing timeout_sec=0.

print("flying on path...")
# 경로를 따라 움직이도록 함. 이 때, 원하는 경로로 움직이는 도중,다른 작업을 계속해서 수행이 가능함.
result = client.moveOnPathAsync([airsim.Vector3r(125,0,z),
                                airsim.Vector3r(125,-130,z),
                                airsim.Vector3r(0,-130,z),
                                airsim.Vector3r(0,0,z)],
                        # velocity : 12, timeout_sec : 120
                        12, 120,
                        # 앞으로만 전진. 드론은 현재 방향을 유지
                        # lookahead : 경로 이동 중에 드론이 다음 위치를 보기 시작하는 거리 -> 20
                        # daptive_lookahead : 경로 이동 중에 드론의 속도에 따라 lookahead 를 자동으로 조정하는 지 여부를 나타내는 값. -> 여기서는 1(True)
                        airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()

# drone will over-shoot so we bring it back to the start point before landing.
client.moveToPositionAsync(0,0,z,1).join()
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")