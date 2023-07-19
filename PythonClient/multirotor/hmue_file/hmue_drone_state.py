import airsim

# AirSim 클라이언트 객체 생성
client = airsim.MultirotorClient()

# AirSim에 연결
client.confirmConnection()

# 드론의 위치 정보 얻기
def get_drone_position():
    # 드론의 현재 위치 정보를 가져옴
    drone_pose = client.simGetVehiclePose()

    # 위치 정보를 X, Y, Z 좌표로 분리
    drone_x = drone_pose.position.x_val
    drone_y = drone_pose.position.y_val
    drone_z = drone_pose.position.z_val

    return drone_x, drone_y, drone_z

if __name__ == "__main__":
    # 드론의 현재 위치 정보 가져오기
    x, y, z = get_drone_position()
    print(f"드론의 현재 위치: X={x}, Y={y}, Z={z}")
