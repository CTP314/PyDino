
import time
from itertools import count
from cv2 import log
from numpy.core.fromnumeric import reshape
import torch
from torch._C import device
import torch.nn  as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
import numpy as np
from win32con import TRUE
from environment import env_init
from environment import get_cactus
import logger

BATCH_SIZE = 50
LR = 0.02
EPSILON = 0.9
GAMMA = 0.99
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 300
env, N_STATES, N_ACTIONS = env_init()
N_S = N_STATES[0] * N_STATES[1] * N_STATES[2]
DEVICE = 'cuda'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(31415)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(N_STATES[0], 6, 5)
        self.conv1.weight.data.normal_(0, 0.1)  
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2.weight.data.normal_(0, 0.1) 
        self.fc1 = nn.Linear(54288, 120)
        self.fc1.weight.data.normal_(0, 0.1)   
        self.fc2 = nn.Linear(120, 24)
        self.fc2.weight.data.normal_(0, 0.1)   
        self.out = nn.Linear(24, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   
    
    def num_flat_fearture(self, x):
        s = 1
        for c in x.shape[1:]:
            s *= c
        return s

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_fearture(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.out(x)
        return action_value

class DQN():
    def __init__(self):
        self.eval_net, self.target_net = Net().to(DEVICE), Net().to(DEVICE)
        print(self.eval_net)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES[0] * N_STATES[1] * N_STATES[2] * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        

    def choose_action(self, x, only_random = True, eps = EPSILON):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(DEVICE)
        if np.random.uniform() < eps and not only_random:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value.cpu(), 1)[1].data.numpy()
            action = action[0]
            print(action_value, action)
        else:
            action = np.random.choice(range(0, N_ACTIONS))
            print("Random this time", action)
        return action

    def store_transition(self, s, a, r, s_):
        s = np.reshape(s, -1)
        s_ = np.reshape(s_, -1)
        transition = np.hstack((s, [a, r], s_))
        
        index = self.memory_counter % MEMORY_CAPACITY
        # print(self.memory_counter, MEMORY_CAPACITY, index)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0 and self.learn_step_counter != 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_S]).to(DEVICE)
        b_a = torch.LongTensor(b_memory[:, N_S:N_S+1].astype(int)).to(DEVICE)
        b_r = torch.FloatTensor(b_memory[:, N_S+1:N_S+2]).to(DEVICE)
        b_s_ = torch.FloatTensor(b_memory[:, -N_S:]).to(DEVICE)

        b_s = torch.reshape(b_s, (-1, N_STATES[0], N_STATES[1], N_STATES[2]))
        b_s_ = torch.reshape(b_s_, (-1, N_STATES[0], N_STATES[1], N_STATES[2]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self):
        torch.save(self.eval_net, 'eval_net.pth')
        torch.save(self.target_net, 'target_net.pth')


if __name__ == '__main__' :

    LOG = logger.Logger('exp_1.txt')
    time.sleep(1)
    print("Please click the left top of the screen.")
    while not env.is_start():
        pass
    time.sleep(1)
    dqn = DQN()
    maxsr = 1
    for i in range(4000):
        LOG.append(f'-------episode {i}-------')
        s = env.reset()
        episode_reward_sum = 0
        nr_sum = 0
        eps = 0.5 if dqn.memory_counter < MEMORY_CAPACITY else EPSILON
        while True:
            a = dqn.choose_action(s, False, eps=eps)
            s_, r, done, c = env.step(a)
            
            nr = (r+c)/2 if not done else -1
            dqn.store_transition(s, a, nr, s_)
            episode_reward_sum += r
            nr_sum += nr
            s = s_
            maxsr = max(episode_reward_sum, maxsr)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if done:
                LOG.append(f"Episode_reward_sum {episode_reward_sum} New_reward_sum {nr_sum} Max_reward_sum {maxsr} #DQN memeory {dqn.memory_counter}")
                break  
        if i%100 == 99:
            dqn.save()
        
    LOG.close()
    dqn.save()

            

# 0 110 770 415