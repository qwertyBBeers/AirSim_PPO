# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from sac_learn import SACagent
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from airgym.envs.hmue_env import AirSimDroneEnv
import numpy as np

def main():

    max_episode_num = 2000 # 최대 에피소드 설정
    env = AirSimDroneEnv("127.0.0.1")
    agent = SACagent(env)  # SAC 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    main()