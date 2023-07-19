import setup_path
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

#드론의 시동을 킴
client.armDisarm(True)
