import setup_path 
import airsim
import time

# This example shows how to use the External Physics Engine
# It allows you to control the drone through setVehiclePose and obtain collision information.
# It is especially useful for injecting your own flight dynamics model to the AirSim drone.

# Use Blocks environment to see the drone colliding and seeing the collision information 
# in the command prompt.

# Add this line to your settings.json before running AirSim:
# "PhysicsEngineName":"ExternalPhysicsEngine"


client = airsim.VehicleClient()
client.confirmConnection()
# client.simGetVehiclePose() : 현재 드론의 pose 값을 들고 옴
pose = client.simGetVehiclePose()

#pose.position 으로 pose의 position 값을 변경시킬 수 있다.
pose.position = airsim.Vector3r(0, 10.0, 0)

#마찬가지로 orientation 값도 변경 가능하다.
pose.orientation = airsim.to_quaternion(0.1, 0.1, 0.1)

# 드론의 위치값을 바꾼다. 이 때, false로 설정을 하면 물리 엔진을 따르지 않고 값을
client.simSetVehiclePose( pose, False )

for i in range(900):
    print(i)
    pose = client.simGetVehiclePose()
    pose.position = pose.position + airsim.Vector3r(0.03, 0, 0)
    pose.orientation = pose.orientation + airsim.to_quaternion(0.1, 0.1, 0.1)
    client.simSetVehiclePose( pose, False )
    time.sleep(0.003)
    
    # 충돌을 얼마나 하였는 지의 정보를 얻을 수 있다.
    collision = client.simGetCollisionInfo()
    if collision.has_collided:
        print(collision)

client.reset()