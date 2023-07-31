import os
import glob
import time
from datetime import datetime

from airgym.envs.hmue_env import AirSimDroneEnv

import torch
import numpy as np

import tensorflow as tf
import gym  

from ppo import PPO

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    print("check")
    env = AirSimDroneEnv("127.0.0.1")
    env_name = "airsim"
    print("통과")

    # 에이전트의 액션 공간이 연속적인지의 여부를 나타내는 변수
    has_continuous_action_space = True  # continuous action space; else discrete

    # 한 에피소드의 최대 타임스텝 수
    max_ep_len = 2000                  # max timesteps in one episode
    
    # 훈련 루프가 종료되는 타임 스텝의 최대 수
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    # 출력 하는 빈도 설정
    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    
    # 평균 보상을 로깅하는 빈도
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    
    # 모델 저장 하는 빈도
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    # 시작하는 액션 분포의 표준 편차
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    
    #action 표준 편차를 감소시킴
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    
    #액션 표준 편차의 최소값
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)

    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    
        #Policy 업데이트를 위해 에이전트가 수집하는 Time step 수
    # update_timestep = max_ep_len * 4      # update policy every n timesteps
    update_timestep = 1000
    # K_epochs = 80               # update policy for K epochs in one PPO update
    # PPO 업데이트에서 사용되는 업데이트 반복 횟수

    K_epochs = 40               

    # PPO 알고리즘에서 사용되는 클리핑 범위
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    # Actor 네트워크 학습률, Critic 네트워크 학습률
    lr_actor = 1e-4       # learning rate for actor network
    # lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 1e-3       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name)

    # state space dimension

    # 신경망에 넣는 input 배열의 크기가 어떤 지
    state_dim = 1006
    # state_dim = env.observation_space.shape[0]

    # action space dimension
    
    # action 으로 나오는 배열의 크기가 어떤 지
    action_dim = 1
    # if has_continuous_action_space:
    #     action_dim = env.action_space.shape[0]
    # else:
    #     action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    # log_dir 이라는 폴더가 없으면 폴더를 만들고 저장
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    tensorboard_log_dir = "tensorboard_logs/"
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_log_dir)
    tensorboard_writer.set_as_default()

    # #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    # #####################################################

    # ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    # ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    # #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, tensorboard_writer)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        
        for t in range(1, max_ep_len+1):
            # select action with policy
            lidar_data = state["lidar"]
            lidar_data /= 20
            non_zero = lidar_data[:, 1] != 0
            lidar_data[non_zero,1] = (lidar_data[non_zero, 1] + 1)/2
            # print("---------checking lidar_data.shape : ")
            # print(lidar_data)

            position_data = state["position"]
            position_data[0] = position_data[0]/33.4
            position_data[1] = position_data[1]/25.35
            position_data = position_data.reshape(1, 2)
            # print(position_data)
            # print("---------checking position_data.shape : ")
            # print(position_data)

            collision_data = np.array([int(state["collision"]), 0])/2
            collision_data = collision_data.reshape(1, 2)
            # print("---------checking collision_data.shape : ")
            # print(collision_data)

            position_state_data = state["position_state"]
            position_state_data[0] = (position_state_data[0]-93)/57
            position_state_data[1] = (position_state_data[1]-13.5)/27.5
            position_state_data = position_state_data.reshape(1, 2)
            # print("---------checking position_state.shape : ")
            # print(position_state_data)

            state_data = np.concatenate([lidar_data, position_data, collision_data, position_state_data], axis=0)
            state_data = state_data.reshape(1,1006)
            # print("---------checking state_data.shape : ")
            # print(state_data.shape)

            # print("++++++++++checking state_data_reshaped.shape : ")
            # print(state_data_reshaped.shape)
            action = ppo_agent.select_action(state_data)
            
            # saving reward and is_terminals

            # restart simulation            
            env.drone.simPause(is_paused=True)  # Pause the simulation
            # if random_flag:
            #   action = np.random.randn() -a.bound~a.bound
            state, reward, done, _ = env.step(action)
            # stop simulation
            env.drone.simPause(is_paused=False)  # Pause the simulation

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update(time_step)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                
                # tensorboard_writer.scalar("Average Reward", log_avg_reward, step=time_step)
                tf.summary.scalar("Average Reward", log_avg_reward, step=time_step)

                tf.summary.flush()

                # log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                # log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                
                tf.summary.scalar("Average Reward (Print)", print_avg_reward, step=time_step)
                tf.summary.flush()

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break
            
        print("---------checking reward : ")
        print(current_ep_reward)
        print("")
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
