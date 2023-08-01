import airsim
import math

# AirSim 클라이언트 생성
client = airsim.MultirotorClient()

# 드론 연결
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# 드론의 현재 상태 정보 얻기
state = client.getMultirotorState()
orientation = state.kinematics_estimated.orientation

# 현재 yaw 값을 Euler 각도로 변환
_, _, current_yaw = airsim.to_eularian_angles(orientation)

# 90도 회전할 yaw 각도 계산
target_yaw = math.radians(70)

# 드론을 지정한 yaw 각도로 회전
client.moveByRollPitchYawZAsync(0, 0, target_yaw, 0, 5)
# coroutine = client.moveByAngleZAsync(0, 0, 0, -target_yaw, 2)
# client.moveByRollPitchYawZAsync()