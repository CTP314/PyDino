from os import name
from numpy.lib.type_check import imag
import win32api
import win32con
from fastsc import grab_screen
import numpy as np
import time
from PIL import Image
from PIL import ImageFilter

X1, Y1, X2, Y2 = 0, 140, 960, 400

def up():
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)

def down():
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)

def release(): 
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)  
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(48, 0, win32con.KEYEVENTF_KEYUP, 0)

def grab():
    obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
    img = Image.fromarray(obs).resize((obs.shape[1]//2, obs.shape[0]//2))
    s = np.array(img)/255
    # s[s<200] = 0
    # s[s>=200] = 1
    # s[s == 1] = 255
    # Image.fromarray(s).show()
    return s


def get_state():
    
    # img.show()
    s1, s2, s3, s4 = grab(), grab(), grab(), grab()
    s = np.stack((s1, s2, s3, s4))
    # print(s.shape)
    # s[s >= 125] = 255
    # s = np.reshape(s, -1)
    return s

def get_cactus():
    obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
    s = obs[0:350,2:70]
    return np.size(s[s < 100])/np.size(s)

class Action_space:
    def __init__(self):
        self.n = 3

    def do(self, a):
        [release, up, down][a]()
        return 1

class Env:
    def __init__(self):
        self.end_state = np.fromfile("end.bin", dtype=np.uint8)
        self.state = get_state()
        self.action_space = Action_space()

    def is_end(self, obs):
        s = np.reshape(obs[150:200,450:510], -1)
        return np.mean(np.abs(self.end_state-s)) <= 10
    
    def step(self, a):
        # print(a)
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        done = self.is_end(obs)
        c = 0
        if done:
            r = -1
        else:
            r = self.action_space.do(a)
            self.state = get_state()
            c = get_cactus()
        return self.state, r, done, c
    
    def reset(self):
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        done = self.is_end(obs)
        while not done:
            obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
            done = self.is_end(obs)
        time.sleep(1)
        up()
        release()
        release()
        time.sleep(4)
        obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
        return get_state()

    def is_start(self):
        (x, y) = win32api.GetCursorPos()
        mouse_state = win32api.GetKeyState(0x01)
        return (mouse_state == -127 or mouse_state == -128) and x <= 770 and 110 <= y and y <= 415
        
def env_init():
    env = Env()
    n_states = env.state.shape
    n_actions = env.action_space.n
    return env, n_states, n_actions

if __name__ == '__main__' :
    # obs = grab_screen(X1, Y1, X2, Y2)[:,:,0]
    # print(get_state(obs).size)
    # print(get_cactus())
    grab()
    get_state()
    # while(True):
        # print(get_state())
    