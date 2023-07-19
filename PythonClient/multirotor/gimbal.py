import setup_path
import airsim

import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# MultirotorClient.wait_key('Press any key to takeoff')
print("Taking off")
client.takeoffAsync().join()
print("Ready")

for i in range(5):
    # X, Y, Z 좌표로 3.5초 동안 이동시킨다.
    client.moveToPositionAsync(float(-50.00), float( 50.26), float( -20.58), float( 3.5))
    time.sleep(6)
    # 드론의 카메라 위치를 설정한다.
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0.5, 0.5, 0.1))
    client.simSetCameraPose("0", camera_pose)
    client.moveToPositionAsync(float(50.00), float( -50.26), float( -10.58), float( 3.5))
    time.sleep(6)
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-0.5, -0.5, -0.1))
    client.simSetCameraPose("0", camera_pose)
