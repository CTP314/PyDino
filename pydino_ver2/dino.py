import torch
from cnn import CNN
import torch
import torchvision.transforms as transforms
from PIL import Image
import win32api
import win32con
from fastsc import Grab
import time

G = Grab(0, 140, 960, 400)
m = torch.load("checkpoints/cnn1")
m.device = "gpu"
m.eval()
t = 0

def predict(x):
    tfm  = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])
    x = torch.unsqueeze(tfm(Image.fromarray(x)), 0)
    with torch.no_grad():
        x = m.forward(x)
    action = ["stay", "up"][x.argmax(dim=-1)]
    return x.argmax(dim=-1)

def up():
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
    time.sleep(0.1)
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)

def down():
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)

def release(): 
    win32api.keybd_event(40, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)  
    win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(48, 0, win32con.KEYEVENTF_KEYUP, 0)

def stay():
    pass    

if __name__ == '__main__':
    print("The game will start after 1s")
    time.sleep(1)
    print("Start!")
    t = time.time()
    while True:
        a = predict(G.grab_screen())
        nt = time.time()
        [stay, up][a]()
        print(f"[fps] {1/(nt-t)} [action] {a}")
        t = nt
  