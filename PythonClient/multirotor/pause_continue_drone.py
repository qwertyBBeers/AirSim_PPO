import setup_path 
import airsim

import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off")
client.moveByVelocityZAsync(0, 0, -20, 8).join()
time.sleep(3)    

for i in range(1, 6):
    print("Starting command to run for 15sec")
    client.moveByVelocityZAsync(-1*i, -1*i, -20-i, 15)
    time.sleep(5) #run
    print("Pausing after 5sec")

    # simpause 시뮬레이터를 일시정지 하는 것
    client.simPause(True)
    time.sleep(5) #paused
    print("Restarting command to run for 7.5sec")
    # 일정 시간 동안 시뮬레이터를 계속해서 수행시키는 것
    client.simContinueForTime(7.5) 
    time.sleep(10)
    print("Finishing rest of the command")
    client.simPause(False)
    time.sleep(15)
    print("Finished cycle")