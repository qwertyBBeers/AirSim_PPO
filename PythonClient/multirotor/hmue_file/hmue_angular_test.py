import airsim
import numpy as np

# AirSim 클라이언트 생성
client = airsim.MultirotorClient()

# 드론 연결
client.confirmConnection()

def get_drone_direction_vector():
    # 드론 상태 정보 가져오기
    state = client.getMultirotorState()

    # 드론 요(Yaw) 각도 가져오기
    yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]

    # 방향 벡터 계산
    x = np.cos(yaw)
    y = np.sin(yaw)
    z = 0.0  # Z 축 방향은 피치와 롤이 고정되어 있으므로 항상 0으로 가정

    direction_vector = np.array([x, y, z])

    return direction_vector

# 드론 방향 벡터 가져오기
direction_vector = get_drone_direction_vector()

# 출력
print("Drone Direction Vector: {}".format(direction_vector))
