import airsim
import numpy as np

# AirSim 클라이언트 생성
client = airsim.MultirotorClient()

# 드론 연결
client.confirmConnection()

def get_drone_direction_vector():
    # 드론 상태 정보 가져오기
    state = client.getMultirotorState()

    # 드론 방향 벡터 가져오기
    orientation = state.kinematics_estimated.orientation

    # 방향 벡터를 NumPy 배열로 변환
    direction_vector = np.array([orientation.x_val, orientation.y_val, orientation.z_val])

    return direction_vector

# 드론 방향 벡터 가져오기
direction_vector = get_drone_direction_vector()

# 벡터의 크기(노름) 계산
vector_magnitude = np.linalg.norm(direction_vector)

# 출력
print("Direction Vector: {}".format(direction_vector))
print("Vector Magnitude: {}".format(vector_magnitude))
