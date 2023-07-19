# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/main/docs/image_apis.md#computer-vision-mode

import setup_path
import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import time
import sys

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")

cameraType = "depth"

# 현재 스크립트를 실행할 때 전달된 커맨드 라인 인자의 리스트 sys.argv[] 는 배열의 정보를 인자로 담는다.
for arg in sys.argv[1:]:
  
  # 변수의 값을 소문자로 바꾸어 저장한다.
  cameraType = arg.lower()

# 카메라 타입을 문자열과 AirSim 이미지 타입으로 매핑한다.
cameraTypeMap = {
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}

# 유효한 카메라 타입이 아니면 종료
if (cameraType not in cameraTypeMap):
  printUsage()
  sys.exit(0)

#카메라 타입을 프린트
print (cameraTypeMap[cameraType])

# AirSim MultirotorClient 객체를 생성
client = airsim.MultirotorClient()

print("Connected: now while this script is running, you can open another")
print("console and run a script that flies the drone and this script will")
print("show the depth view while the drone is flying.")

help = False

#FPS(Frames Per Second)를 표시하기 위한 폰트 및 변수를 설정
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

# AirSim 시뮬레이터에서 카메라 이미지를 가져와 OpenCV를 사용하여 화면에 표시
while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
    if (rawImage == None):
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        cv2.putText(png,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
        cv2.imshow("Depth", png)

    frameCount = frameCount  + 1
    endTime = time.time()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x')):
        break
