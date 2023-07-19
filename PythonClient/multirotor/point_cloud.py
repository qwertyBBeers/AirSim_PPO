# use open cv to create point cloud from depth image.
import setup_path 
import airsim

import cv2
import time
import sys
import math
import numpy as np

############################################
########## This is work in progress! #######
############################################

# file will be saved in PythonClient folder (i.e. same folder as script)
# point cloud ASCII format, use viewers like CloudCompare http://www.danielgm.net/cc/ or see http://www.geonext.nl/wp-content/uploads/2014/05/Point-Cloud-Viewers.pdf
outputFile = "cloud.asc" 
color = (0,255,0)
rgb = "%d %d %d" % color

# 4X4 투영 행렬 정의. 이미지 좌표를 3D 공간 좌표로 변환하는 데 사용
projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
                              [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                              [0.000000000, 0.000000000, 10.00000000, 100.00000000],
                              [0.000000000, 0.000000000, -10.0000000, 0.000000000]])

# 사용법을 출력하는 함수 정의
def printUsage():
   print("Usage: python point_cloud.py [cloud.txt]")
  
# 변환된 포인트 클라우드를 파일에 저장
def savePointCloud(image, fileName):
   
   f = open(fileName, "w")
   # 모든 픽셀에 대해서 반복
   for x in range(image.shape[0]):
     for y in range(image.shape[1]):
        pt = image[x,y]
   # pt[0]의 값이 무한대(inf) 이거나 NaN인지 확인. Depth 이미지에서 유효한 값이 아니니 skip함
        if (math.isinf(pt[0]) or math.isnan(pt[0])):
          # skip it
          None
        else: 
          # 좌표와 색상 정보를 포인트 클라우드 파일에 기록
          f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, rgb))
   f.close()

# txt 파일로 저장
for arg in sys.argv[1:]:
  cloud.txt = arg

client = airsim.MultirotorClient()

while True:
    # Airsim으로부터 Depth 이미지를 가져옴
    rawImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
    # 만약 이미지를 가져 오지 못했을 경우
    if (rawImage is None):
        print("Camera is not returning image, please check airsim for error messages")
        airsim.wait_key("Press any key to exit")
        sys.exit(0)
    # 중지한다.

    else:
        # 가져온 이미지 데이터를 디코딩하여 PNG 형식으로 변환
        png = cv2.imdecode(np.frombuffer(rawImage, np.uint8) , cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
        Image3D = cv2.reprojectImageTo3D(gray, projectionMatrix)
        savePointCloud(Image3D, outputFile)
        print("saved " + outputFile)
        airsim.wait_key("Press any key to exit")
        sys.exit(0)

    key = cv2.waitKey(1) & 0xFF;
    if (key == 27 or key == ord('q') or key == ord('x')):
        break;
