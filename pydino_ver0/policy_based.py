import numpy as np
import cv2
from torch._C import device
from fastsc import grab_screen
import time
import win32api
import win32con
from itertools import count
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count

class FCN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n_states, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class PolicyGradient:
    def __init__(self, n_states, n_actions, device='cpu', gamma=0.01, lr=0.01):
        self.gamma = gamma
        self.device = device
        self.policy_net = FCN(n_states, n_actions).to(device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action
    
    def update(self, reward_pool, state_pool, action_pool):
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] < -1:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add
        
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)

        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        

        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = torch.FloatTensor([action_pool[i]]).to(self.device)
            reward = reward_pool[i]
            
            state = torch.from_numpy(state).float().to(self.device)
            probs = self.policy_net(state)
            m = Categorical(probs)
            loss = -m.log_prob(action) * reward
            loss.backward()
        
        self.optimizer.step()

X1, Y1, X2, Y2 = 0, 140, 960, 370

def jump():
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)

def down():
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)

def up():
    win32api.keybd_event(48, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)

def stay():
    pass

def is_start():
    (x, y) = win32api.GetCursorPos()
    mouse_state = win32api.GetKeyState(0x01)
    # print(mouse_state, x, y, (mouse_state == -127 or mouse_state == -128) and x <= 770 and 110 <= y and y <= 415)
    return (mouse_state == -127 or mouse_state == -128) and x <= 770 and 110 <= y and y <= 415


class Env:
    def __init__(self):
        self.end_state = np.fromfile("end.bin", dtype=np.uint8)
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        self.state = np.reshape(obs[:,0:450], -1)
        self.action_space = [jump, down, stay]

    def is_end(self, obs):
        s = np.reshape(obs[150:200,450:510], -1)
        return np.mean(np.abs(self.end_state-s)) <= 10
    
    def step(self, action):
        self.action_space[action]()
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        self.state = np.reshape(obs[:,0:450], -1)
        done = self.is_end(obs)
        reward = 0.9
        if done:
            reward = -2
        elif action == 2:
            reward = 2
        print(["jump", "down", "stay"][action])
        return self.state, reward, done
    
    def reset(self):
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        done = self.is_end(obs)
        while not done:
            obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
            done = self.is_end(obs)
        time.sleep(1)
        up()
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        return np.reshape(obs[:, 0:450], -1)
        
            
        

def env_init():
    env = Env()
    n_states = env.state.shape[0]
    n_actions = len(env.action_space)
    return env, n_states, n_actions


if __name__ == '__main__' : 
    # last_time = time.time()
    while(True):
        state = grab_screen(X1, Y1, X2, Y2)[:,0:450,0]

        cv2.imshow('windows', state)
              

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    # env.reset()


    time.sleep(1)
    print("Please click the left top of the screen.")
    while is_start() is not True:
        pass
    
    env, n_states, n_actions = env_init()

    print(env, n_states, n_actions)

    agent = PolicyGradient(n_states, n_actions, device = 'cuda', lr = 0.05)

    state_pool = []
    action_pool = []
    reward_pool = []

    print("Resetting the environment")

    for i_episode in range(1200):
        state = env.reset()
        if i_episode == 0:
            print("Begin training")
        ep_reward = 0
        for t in count():
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:', i_episode, 'Reward:', ep_reward)
                break
        if i_episode > 0 and i_episode % 5 == 4:
            agent.update(reward_pool, state_pool, action_pool)
            state_pool = []
            action_pool = []
            reward_pool = []  
            print("Finished update")
            

# 0 110 770 415