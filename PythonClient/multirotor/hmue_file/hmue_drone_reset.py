import airsim

def set_robot_pose(x, y, z, pitch, roll, yaw):
    # AirSim에 연결
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 로봇 위치 설정
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))
    client.simSetVehiclePose(pose, ignore_collision=True)
    client.simSetVelocity(0,0,0)
if __name__ == "__main__":
    # 원하는 로봇 위치 및 방향 설정 (예시: x=0, y=0, z=-10, pitch=0, roll=0, yaw=0)
    desired_position = (0, 0, -10)
    desired_orientation = (0, 0, 0)

    set_robot_pose(*desired_position, *desired_orientation)
