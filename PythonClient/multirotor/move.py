import airsim
import pygame
from pygame.locals import *

# AirSim 연결 설정
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# MultirotorClient.wait_key('Press any key to takeoff')
print("Taking off")
client.takeoffAsync().join()
print("Ready")

# Pygame 초기화
pygame.init()
window_size = (400, 400)
window = pygame.display.set_mode(window_size)

# 드론 움직임 매핑
key_mapping = {
    K_UP: (airsim.Vector3r(0, 1, 0), '앞으로'),
    K_DOWN: (airsim.Vector3r(0, -1, 0), '뒤로'),
    K_LEFT: (airsim.Vector3r(-1, 0, 0), '왼쪽'),
    K_RIGHT: (airsim.Vector3r(1, 0, 0), '오른쪽'),
}

# 주요 제어 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # 현재 눌려진 키 확인
    keys_pressed = pygame.key.get_pressed()

    # 화살표 키 입력 확인
    for key, (direction, movement) in key_mapping.items():
        if keys_pressed[key]:
            # 드론을 지정된 방향으로 이동시킴
            client.moveByVelocityAsync(
                direction.x_val, direction.y_val, direction.z_val, duration=0.1
            )

            # 이동 방향 출력
            print(f"{movement} 방향으로 이동 중")

    # 창 업데이트
    pygame.display.flip()

# 정리 작업
pygame.quit()
