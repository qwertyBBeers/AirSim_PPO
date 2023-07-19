import airsim
from datetime import datetime

'''
Simple script with settings to create a high-resolution camera, and fetching it

Settings used-
{
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "Vehicles" : {
        "Drone1" : {
            "VehicleType" : "SimpleFlight",
            "AutoCreate" : true,
            "Cameras" : {
                "high_res": {
                    "CaptureSettings" : [
                        {
                            "ImageType" : 0,
                            "Width" : 4320,
                            "Height" : 2160
                        }
                    ],
                    "X": 0.50, "Y": 0.00, "Z": 0.10,
                    "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
                },
                "low_res": {
                    "CaptureSettings" : [
                        {
                            "ImageType" : 0,
                            "Width" : 256,
                            "Height" : 144
                        }
                    ],
                    "X": 0.50, "Y": 0.00, "Z": 0.10,
                    "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
                }
            }
        }
    }
}
'''

# AirSim 라이브러리를 사용하여 드론의 고해상도 카메라와 저해상도 카메라로 이미지를 캡처하는 간단한 스크립트
client = airsim.VehicleClient()
client.confirmConnection()

# 현재까지 캡쳐한 프레임 수를 저장하는 변수
framecounter = 1

# 이전 타임스탬프를 저장하는 변수. 30 프레임마다 경과 시간을 계산하는 데 사용하고 있다.
prevtimestamp = datetime.now()

while(framecounter <= 500):
    if framecounter%150 == 0:

        # 고해상도 카메라로 이미지를 캡처한다.
        client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
        print("High resolution image captured.")

    if framecounter%30 == 0:
        now = datetime.now()
        print(f"Time spent for 30 frames: {now-prevtimestamp}")
        prevtimestamp = now

    # 저해상도 카메라로 이미지를 캡처한다.
    client.simGetImages([airsim.ImageRequest("low_res", airsim.ImageType.Scene, False, False)])
    framecounter += 1
