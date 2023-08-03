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
parser.add_argument('--env-name', default="airsim",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--max_episode_steps', default="2000")
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = AirSimDroneEnv("127.0.0.1")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#action_space 설정

# 환경 차원 정의
observation_space = 3140

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
memory = ReplayMemory(args.replay_size, args.seed)

#CNN 구축
cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

# Training Loop
total_numsteps = 0
updates = 0
reward_val = 0
episode_val = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False

    state = env.reset()
    
    camera = torch.tensor(state["camera"])
    camera = torch.unsqueeze(camera, 0)
    camera = torch.unsqueeze(camera, 0)
    camera = camera.float()
    features = cnn(camera) # state --> image data        
    
    #torch.Size([1, 3136])
    state["position"][0] = (state["position"][0]+23.6)/57
    state["position"][1] = (state["position"][1]+2.15)/27.5
    position = state["position"].reshape(1,2)

    state["position_state"][0] = (state["position_state"][0]-93)/57
    state["position_state"][1] = (state["position_state"][1]-13.5)/27.5
    position_state = state["position_state"].reshape(1,2)
    
    position = torch.tensor(position)
    position_state = torch.tensor(position_state)
    
    state = torch.cat((features, position, position_state), dim=1)
    state = state.detach().numpy()
    
    while not done:
        
        if args.start_steps > total_numsteps:
            action = action_space.sample()  # Sample random action
        else:
            # print(state.shape)
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                
                # Update parameters of all the networks

                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        
        next_state, reward, done, _ = env.step(action) # Step

        camera = torch.tensor(next_state["camera"])
        camera = torch.unsqueeze(camera, 0)
        camera = torch.unsqueeze(camera, 0)
        camera = camera.float()
        features = cnn(camera) # state --> image data        
        
        #torch.Size([1, 3136])
        next_state["position"][0] = (next_state["position"][0]+23.6)/57
        next_state["position"][1] = (next_state["position"][1]+2.15)/27.5
        position = next_state["position"].reshape(1,2)

        next_state["position_state"][0] = (next_state["position_state"][0]-93)/57
        next_state["position_state"][1] = (next_state["position_state"][1]-13.5)/27.5
        position_state = next_state["position_state"].reshape(1,2)
        position = torch.tensor(position)
        position_state = torch.tensor(position_state)
        
        next_state = torch.cat((features, position, position_state), dim=1)
        next_state = next_state.detach().numpy()
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        reward_val += reward
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == 2000 else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break
    
    
    
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        reward_val = reward_val / 10
        writer.add_scalar('reward/avg', reward_val, i_episode)
        avg_reward = 0.
        episodes = 1
        for _  in range(episodes):
            
            state = env.reset()

            camera = torch.tensor(state["camera"])
            camera = torch.unsqueeze(camera, 0)
            camera = torch.unsqueeze(camera, 0)
            camera = camera.float()
            features = cnn(camera) # state --> image data        
            
            #torch.Size([1, 3136])
            state["position"][0] = (state["position"][0]+23.6)/57
            state["position"][1] = (state["position"][1]+2.15)/27.5
            position = state["position"].reshape(1,2)

            state["position_state"][0] = (state["position_state"][0]-93)/57
            state["position_state"][1] = (state["position_state"][1]-13.5)/27.5
            position_state = state["position_state"].reshape(1,2)
            position = torch.tensor(position)
            position_state = torch.tensor(position_state)
            
            state = torch.cat((features, position, position_state), dim=1)
            state = state.detach().numpy()

            episode_reward = 0
            done = False
            while not done:             
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                camera = torch.tensor(next_state["camera"])
                camera = torch.unsqueeze(camera, 0)
                camera = torch.unsqueeze(camera, 0)
                camera = camera.float()
                features = cnn(camera) # state --> image data        
                
                #torch.Size([1, 3136])
                next_state["position"][0] = (next_state["position"][0]+23.6)/57
                next_state["position"][1] = (next_state["position"][1]+2.15)/27.5
                position = next_state["position"].reshape(1,2)

                next_state["position_state"][0] = (next_state["position_state"][0]-93)/57
                next_state["position_state"][1] = (next_state["position_state"][1]-13.5)/27.5
                position_state = next_state["position_state"].reshape(1,2)
                position = torch.tensor(position)
                position_state = torch.tensor(position_state)
                
                next_state = torch.cat((features, position, position_state), dim=1)

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

