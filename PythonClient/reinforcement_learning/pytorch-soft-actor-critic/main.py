import argparse
import datetime
import gym
from gym import spaces
import numpy as np
import itertools
import torch
from sac import SAC
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from airgym.envs.hmue_env import AirSimDroneEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

# 환경에 대한 설명. 어떤 환경의 이름을 쓸 지 설정한다.
parser.add_argument('--env-name', default="airsim",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
# policy의 유형을 지정하는 문자열 인수
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
# 최대 episode_step의 개수
parser.add_argument('--max_episode_steps', default="2000")

#평가를 진행할 때 사용하는 부분
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')

# discount factor(감가율)에 대한 설정
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

# 타겟 네트워크를 업데이트 시 사용되는 값 지정
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')

# learning rate (학습률) 지정
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

# 엔트로피와 보상 간 상대적 중요성을 결정하는 매개변수
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')

# 자동으로 엔트로피 조절을 사용할 지 여부를 지정하는 부울 함수
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')

# 랜덤 시드를 지정하는 정수 인수 
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')

# 배치 크기를 지정. 저장하는 크기를 지정하는 곳
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

# 최대 시뮬레이션 스텝 수를 지정. max_episode_steps와 같은 역할
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')

# hidden layer 층에 대한 설정.
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

# 스텝 당 모델이 업데이트 하는 수를 지정
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

# 무작위 액션을 샘플링하여 스텝의 데이터를 모집하는 정수
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')

# value target 업데이트 주기를 지정
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

# replay buffer 크기를 지정함. step 수와 비슷
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# 환경 지정. 학습 환경은 따로 제작되어져야 함
env = AirSimDroneEnv("127.0.0.1")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#action_space 설정  

# 환경 차원 정의
# input으로 들어가는 차원을 정의
observation_space = 3256

# 연속적인 행동 공간 정의
low = -1.0
high = 1.0
action_space = spaces.Box(low=low, high=high, shape=(1,1))

# Agent
#state_dim, action_dim 넣기
agent = SAC(observation_space, action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory

# 에이전트가 과거의 경험을 저장하고 재사용하여 학습하는 데 사용. 학습 데이터의 상관성을 줄이고, 학습의 안정성을 높일 수 있음.
memory = ReplayMemory(args.replay_size, args.seed)

# CNN 구축
# in_channels는 input로 들어가는 data의 채널을 의미. 환경에서 grayscale을 받아 와서 1이다.
# kernel_size는 컨볼루션 커널(output)의 크기이다.
# out_channels는 출력하는 배열의 크기를 나타낸다.
# stride는 필터의 이동 간격이다. 한 번에 4픽셀 씩 이동하며 컨볼루션을 수행한다.

# CNN은 conv layer가 3개이고, 나오게 되는 채널이 64인 신경망이다.

# Training Loop
total_numsteps = 0
updates = 0
reward_val = 0
episode_val = 0

# 무한 루프
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False

    # 환경을 state로 불러온다.
    state = env.reset()
    
    # CNN을 적용시킴. 84X84의 grayscale image를 넣어주었다.
    # camera = torch.tensor(state["camera"])
    # camera = torch.unsqueeze(camera, 0)
    # camera = torch.unsqueeze(camera, 0)
    # camera = camera.float()
    # features = self.cnn(camera) # state --> image data        
    #torch.Size([1, 3136]) -> feature의 크기는 다음과 같았다.
    
    # Nomalize 진행. 모든 state 값을 0~1 사이로 정규화 진행하였다.
    state = agent.cnn_nomalize(state)
    #input이 numpy로 진행되어야 하기에 numpy 진행
    state = state.detach().numpy()
    
    while not done:
        if args.start_steps > total_numsteps:
            # 일정 step 이전에는 exploration을 진행하며 데이터 수집
            action = action_space.sample()  # Sample random action
        else:
            # 일정 step 이후 policy 에 맞는 action 진행 
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            # memory가 다 차게 된다면
            for i in range(args.updates_per_step):
                # 기록 진행
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        
        # action을 통해 나온 값을 저장한다.
        next_state, reward, done, _ = env.step(action) # Step
        # action에 맞는 reward, 다음 state 정보 등을 불러옴
        
        #next_state에 대한 Nomalize 진행
        # camera = torch.tensor(next_state["camera"])
        # camera = torch.unsqueeze(camera, 0)
        # camera = torch.unsqueeze(camera, 0)
        # camera = camera.float()
        # features = cnn(camera) # state --> image data        
        
        #torch.Size([1, 3136])
        next_state = agent.cnn_nomalize(next_state)
        next_state = next_state.detach().numpy()
        
        # episode와 reward 기록
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        reward_val += reward
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        
        mask = 1 if episode_steps == 2000 else float(not done)
        
        #episode 2000개 씩 메모리에 저장
        memory.push(state, action, reward, next_state, mask) # Append transition to memory
        

        #state에 다음 state 정보를 저장
        state = next_state
        

    # max_step 진행 이후 끝낸다.
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # 일정 episode(10회 씩) 이후 평가 진행
    if i_episode % 10 == 0 and args.eval is True:
        reward_val = reward_val / 10
        writer.add_scalar('reward/avg', reward_val, i_episode)
        avg_reward = 0.
        episodes = 1
        for _  in range(episodes):
            
            state = env.reset()

            # camera = torch.tensor(state["camera"])
            # camera = torch.unsqueeze(camera, 0)
            # camera = torch.unsqueeze(camera, 0)
            # camera = camera.float()
            # features = cnn(camera) # state --> image data        
            
            #torch.Size([1, 3136])
            state = agent.cnn_nomalize(state)
            state = state.detach().numpy()

            episode_reward = 0
            done = False
            while not done:             
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                next_state = agent.cnn_nomalize(next_state)
                state = next_state.detach().numpy()
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

