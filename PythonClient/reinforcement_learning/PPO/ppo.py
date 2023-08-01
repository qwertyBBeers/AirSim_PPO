import torch
import tensorflow as tf
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
# cuda 를 사용하고, 그러지 않을 시 cpu를 사용한다.
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
# 
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    # RollouBuffer 객체의 내부 데이터를 초기화 하는 역할

    def clear(self):
        del self.actions[:] # actions : 수헹된 액션들을 저장하는 리스트
        del self.states[:] # states : 관찰된 상태들을 저장하는 리스트
        del self.logprobs[:] # logprobs : 액션의 로그 확률들을 저장하는 리스트
        del self.rewards[:] # rewards : 받은 보상들을 저장하는 리스트
        del self.state_values[:] # 상태의 가치(V)를 저장하는 리스트
        del self.is_terminals[:] # 에피소드의 종료 여부를 저장하는 리스트

# 신경망 기반의 액터 크리틱 모델을 구현, 주어진 상태를 기반으로 액션을 선택, 선택한 액션의 로그 확률과 상태의 가치를 예측
class ActorCritic(nn.Module):
    # state_dim : 상태의 차원 값, action_dim : action의 차원 값, has_continuous_action_space : 연속적인 액션 공간을 가지는 지 여부를 나타내는 Bool 값. action_std_init : 액션의 분산을 초기화
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        #nn.Module의 생성자를 호출하여 초기화 / super()는 상속 시 부모 클래스의 매서드를 호출할 때 사용
        super(ActorCritic, self).__init__()
        
        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            # action_dim은 연속적인 행동 공간의 차원 수를 나타낼 것이고
            self.action_dim = action_dim
            # action_var은 행동공간 내에서의 분산을 나타낸다. 이 줄은 action_dim 차원의 텐서를 생성하고, 각 차원의 값을 action_std_init**2 로 설정한다. 이로 모든 차원의 분산값을 초기화 한다.
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor

        # # has_continuous_action_space 값이 연속적인 액션 공간을 가질 때, 즉 연속적인 행동 공간을 다루고 있다면
        if has_continuous_action_space :
            # nn.Sequential : 연속적인 레이어를 순차적으로 적용하는 신경망 모델을 정의하는 PyTorch의 클래스
            self.actor = nn.Sequential(
                            # 입력으로 주어진 상태(State)를 64차원의 벡터로 변환하는 선형 레이어
                            nn.Linear(state_dim, 64),
                            # Hyperbolig Tangent 함수를 적용 -> 활성화 함수이다. 신경망에서 중간 계층에서 사용되며 비선형성을 도입하여 네트워크가 더 복잡한 함수를 모델링 할 수 있도록 도와줌
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            #64차원의 벡터값을 action값으로 변경해 줌
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        # has_continuous_action_space 값이 연속적인 액션 공간을 가지지 않을 때, 즉 이산적인(discrete한) 행동 공간을 다루고 있다면
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        # (Value Function)을 나타내는 신경망이다. 함수는 주어진 상태에서 예상되는 보상의 총합을 추정하는 역할. -> Critic 역할 수행
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            #초기화 했던 분산의 값을 설정해 준다.
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            # ActorCritic 의 분산이 적용되지 않고, discrete한 값을 지닌다는 것을 표현
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")
    # forward() => 추상 메서드 : 파이썬에서 메서드의 선언만 한 것. 이 부분은 신경망의 순전파 연산을 한 부분인데, 이 뜻은 입력 데이터를 입력으로 받아 계산 그래프를 통해 출력을 계산하는 수행을 하는 것을 말한다.
    def forward(self):
        raise NotImplementedError
    
    # 주어진 상태 (State)에 따른 행동(action)을 출력
    def act(self, state):

        if self.has_continuous_action_space:    
            # state에 대한 action을 받음
            action_mean = self.actor(state)
            
            # 행동에 대한 공분산 행렬을 생성하는 부분
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # cov_mat = torch.diag_embed(self.action_var).expand(action_mean.size(0), -1, -1)


            # 평균과 공분산 행렬을 입력으로 받아 다변수 정규분포 생성
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # 정의된 확률에 따라 행동을 샘플링
        action = dist.sample()
        
        # 최종적으로 선택된 행동에 대한 log 확률을 계산
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        # 최종적으로 선택된 행동을 반환
        return action.detach(), action_logprob.detach(), state_val.detach()
        # .detach() : 파이토치에서 텐서를 복사하여 새로운 텐서를 생성하는 메서드. 이는 그래디언트를 계산X, 역전파 시에 업데이트X

    # 상태와 선택한 액션에 대한 평가    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:

            # 상태에 대한 행동 저장
            action_mean = self.actor(state)
            
            # action_var에는 기존의 차원과 동일한 크기로 확장된 값이 저장됨. action_var은 연속적인 action 공간에서 사용되는 액션 분산인데, 이 값을 action_mean 값과 같게 확장하여 액션 차원에 대한 분산값을 갖는 텐서로 만듦
            # expand_as : 동일한 크기로 확장
            action_var = self.action_var.expand_as(action_mean)
            # action_var를 대각행렬로 만들고 차원을 확장한 결과 저장
            cov_mat = torch.diag_embed(action_var).to(device)

            # dist는 확률분포객체를 말한다. multivariatenomal은 다변량 정규분포를 말하는데, 즉 dist는 액션 평균과 분산을 가지는 다변량 정규 분포이다.
            # 다변량 정규 분포란 여러 개의 확률 분포를 함께 다루는 확률 분포
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                # 차원 크기가 1이면, 차원의 크기를 자동으로 재구성하게 함
                action = action.reshape(-1, self.action_dim)
        else:
            # 액션이 discrete 하다면 액션의 확률 분포를 계산
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        # dist_entropy는 말 그대로 dist의 엔트로피 값을 의미하며, 이는 확률 분포의 불확실성을 나타내는 지표이다.
        dist_entropy = dist.entropy()
        #상태가치함수의 값을 나타냄
        state_values = self.critic(state)
        
        #최종적인 State_value와 불확실성 반환
        return action_logprobs, state_values, dist_entropy

# PPO 알고리즘 구현
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init, tensorboard_writer):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        
        self.tensorboard_writer = tensorboard_writer

        # Adam 을 사용하여 옵티마이저 설정.여기서 [파라미터 초기화 -> 그래디언트 계산 -> 파라미터 업데이트]를 반복한다.
        self.optimizer = torch.optim.Adam([
                        # self.policy.actor는 actor의 신경망의 파라미터들을 나타냄
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        # self.policy.critic.parameters()는 critic 신경망의 파라미터들을 나타냄.
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                        # lr은 학습률이며, 이 값들로 파라미터 업데이트 속도를 조절함.
                    ])

        # PPO의 주 점 : 이전 Policy들을 저장해 놓는다.PPO 알고리즘은 현재 정책과 이전 정책을 사용하여 신경망 파라미터를 업데이트 한다.
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # PPO 클래스의 생성자에 사용. 평균제곱오차 손실 함수를 나타낸다. 이 부분에서는 예측값과 타겟값 사이의 평균 제곱 오차를 계산한다.
        # 즉, error 값을 의미한다.
        self.MseLoss = nn.MSELoss()

    # PPO 클래스의 메서드. new_action_std 값을 기반으로 정책 신경망의 액션 분포의 표준 편차를 설정
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            # 새로운 action_std 값을 action_std 안에 저장
            self.action_std = new_action_std
            # 새로운 값을 이용하여 policy와 policy_old 간의 표준 편차를 업데이트
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    # action 사이의 표준 편차를 감소 => CLIP이 생기는 부분
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            # 변화된 action을 업데이트
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    # 주어진 상태에 대한 action을 선택하고 기록
    def select_action(self, state):

        if self.has_continuous_action_space:
            # with torch.no_grad()는 그래디언트를 계산하지 않는 연산 실행
            with torch.no_grad():
                # 상태를 torch.FloatTensor로 변환하고 디바이스에 전송. 이후 이전 policy의 모델을 사용해 상태로부터 액션, 액션로그확률, 상태가치를 샘플링함
                #즉, state를 보낸 이후, 그 상태에 대한 action 선택
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()
    
    # 정책을 업데이트 하는 역할. 에이전트가 수집한 경험 데이터를 기반으로 정책을 최적화하여 학습
    def update(self, time_step):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        # PPO 알고리즘에서 Reward를 계산하는 부분
        # self.buffer.reward는 에이전트가 수집한 경험 데이터의 보상값, self.buffer.is_terminals는 종료 여부 정보 포함.
        # reverse 함수를 이용해서 처음으로 거슬러 가서 시작.
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            # 터미널이 종료 되었는가
            if is_terminal:
                # 에피소드가 종료 되었으면, 할인된 보상을 0으로 다시 초기화함
                discounted_reward = 0
            
            # discounted_reward(현재 리워드)는 기존 리워드 + 감마*Reward 이다.
            discounted_reward = reward + (self.gamma * discounted_reward)
            # 리스트의 앞 부분에 현재 리워드를 입력한다. 제일 앞 부분에 리워드를 입력
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        # 리워드를 정규화 하는 과정. PPO 에서는 리워드를 정규화 해 주어 학습의 안정성을 높임
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # 기존 정보를 불러옴
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        #현재 리워드와 과거 리워드에 대해서 계산. 이후, 보상의 차이를 통해 정책의 개선 방향 결정
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        # PPO 알고리즘의 정책을 업데이트를 수행하는 부분 주어진 에포크 수 ( K_epochs ) 만큼 반복하면서 다음을 수행
        for _ in range(self.K_epochs):  

            # Evaluating old actions and values
            # 기존의 state와 action을 평균, 저장
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            #현재와 이전 정책의 비율을 계산
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # surr1 : 비율과 보상간 차이의 곱, surr2 : 비율을 CLIP 한 값과 보상간 차이의 곱
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # 손실에 대한 계산
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        # logging
        tf.summary.scalar("loss", loss.mean().detach().cpu().numpy(), step=time_step)

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


