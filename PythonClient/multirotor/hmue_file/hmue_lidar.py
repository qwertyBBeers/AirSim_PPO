import airsim
import numpy as np
import matplotlib.pyplot as plt

def lidar_to_image():
    # AirSim에 연결
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # LIDAR 데이터 수집
    lidar_data = client.getLidarData()
    points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    # 이미지 크기 및 범위 설정
    image_size = 256  # 이미지 크기 (가로, 세로 픽셀 수)
    max_range = 30.0  # LIDAR에서 최대로 측정 가능한 거리 (meters)

    # 이미지 생성 및 초기화 (배경을 흰색으로 설정)
    image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # 베셀 범위 이미징 수행
    for point in points:
        x, y, z = point
        if 0 < z < max_range:  # 최대 거리 이내의 점만 이미지에 표시
            pixel_x = int((x / max_range + 0.5) * image_size)
            pixel_y = int((y / max_range + 0.5) * image_size)

            # 이미지 범위를 벗어나는 좌표 값을 제한
            pixel_x = np.clip(pixel_x, 0, image_size - 1)
            pixel_y = np.clip(pixel_y, 0, image_size - 1)

            image[pixel_y, pixel_x] = 0  # 점의 위치를 검은색으로 표시

    return image

if __name__ == "__main__":
    lidar_image = lidar_to_image()

    # 이미지를 표시
    plt.imshow(lidar_image, cmap='gray')
    plt.axis('off')
    plt.show()
