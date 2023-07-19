import airsim
import os

# connect to the AirSim simulator
# MultirotorClient 객체를 생성하여 client 변수에 할당한다. 이 객체는 통신을 관리하고, 드론을 제어하는 기능을 제공한다.
client = airsim.MultirotorClient()

# client 객체를 이용해 AirSim 시뮬레이터와의 연결을 확인한다.
client.confirmConnection()

# 드론의 API 제어를 활성화한다. 이로 인해 코드를 통해 드론을 제어할 수 있다.
client.enableApiControl(True)

#드론의 시동을 건다.
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
#드론을 이륙한다. join 메서드를 사용하면 작업이 완료될 때 까지 기다린다.
client.takeoffAsync().join()
#어느 포지션으로 이동시키는 것을 의미한다.
client.moveToPositionAsync(-30, 10, -10, 5).join()
