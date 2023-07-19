    """
    For connecting to the AirSim drone environment and testing API functionality
    """
    import setup_path 
    import airsim

    import os
    import tempfile
    import pprint

    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 드론의 상태를 들고온다
    state = client.getMultirotorState()
    
    # 상태 정보를 문자열로 변환하여 출력한다.
    s = pprint.pformat(state)
    print("state: %s" % s)

    # 드론을 수동 모드로 변환하여 조종한다.
    client.moveByManualAsync(vx_max = 1E6, vy_max = 1E6, z_min = -1E6, duration = 1E10)
    airsim.wait_key('Manual mode is setup. Press any key to send RC data to takeoff')

    # RC 데이터를 보내서 드론을 조종할 수 있다.
    client.moveByRC(rcdata = airsim.RCData(pitch = 0.0, throttle = 1.0, is_initialized = True, is_valid = True))

    airsim.wait_key('Set Yaw and pitch to 0.5')

    # RC 데이터를 보내 움직이는 장면
    client.moveByRC(rcdata = airsim.RCData(roll = 0.5, throttle = 1.0, yaw = 0.5, is_initialized = True, is_valid = True))
