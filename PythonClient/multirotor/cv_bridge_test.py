import rospy
from sensor_msgs.msg import Image,CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import airsim
import cv2
import numpy as np

CLAHE_ENABLED = False  # when enabled, RGB image is enhanced using CLAHE

CAMERA_FX = 320
CAMERA_FY = 320
CAMERA_CX = 320
CAMERA_CY = 240

CAMERA_K1 = -0.000591
CAMERA_K2 = 0.000519
CAMERA_P1 = 0.000001
CAMERA_P2 = -0.000030
CAMERA_P3 = 0.0

IMAGE_WIDTH = 640  # resolution should match values in settings.json
IMAGE_HEIGHT = 480


img = cv2.imread("test.png")
bridge = CvBridge()
img_msg = bridge.cv2_to_imgmsg(img)
print(img_msg.encoding)