from os import name
from numpy.lib.type_check import imag
import win32api
import win32con
from fastsc import Grab
import numpy as np
import time
from PIL import Image
from PIL import ImageFilter
import cv2

G = Grab(0, 140, 960, 400)

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

def is_start():
    (x, y) = win32api.GetCursorPos()
    mouse_state = win32api.GetKeyState(0x01)
    return (mouse_state == -127 or mouse_state == -128) and x <= 770 and 110 <= y and y <= 415

if __name__ == '__main__':
    last_time = time.time()
    st = [0, 0]
    while(True):
        s = G.grab_screen()
        cv2.imshow('windows', s)
        up()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
