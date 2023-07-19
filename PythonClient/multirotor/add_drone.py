import setup_path 
import airsim
import tempfile
import os
import numpy as np
import cv2
import pprint

# connect to the AirSim simulator
client = airsim.MultirotorClient()

#AirSim과의 연결 확인
client.confirmConnection()

# add new vehicle
vehicle_name = "Drone2"
# 새로운 드론의 위치 입력
pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))

# 새로운 드론을 시뮬레이터에 추가
client.simAddVehicle(vehicle_name, "simpleflight", pose)

# 드론의 API 제어를 활성화
client.enableApiControl(True, vehicle_name)

# 드론의 시동 걸기
client.armDisarm(True, vehicle_name)

# 드론 이륙. 10.0 정도 이륙시키기
client.takeoffAsync(10.0, vehicle_name)

# requests = [] : 이미지를 요청하는 데 사용되는 객체들의 리스트
requests = [airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]  #scene vision image in uncompressed RGBA array

# AirSim 시뮬레이터로부터 이미지 응답을 받음
responses = client.simGetImages(requests, vehicle_name=vehicle_name)
print('Retrieved images: %d' % len(responses))

# 임시 저장소 디렉토리 지정 이 곳에 image를 저장
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx, response in enumerate(responses):
    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    else:
        print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)