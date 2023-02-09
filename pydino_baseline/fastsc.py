import win32gui
import win32ui
import win32con
import win32api
import cv2
import numpy as np
import time

hdesktop = win32gui.GetDesktopWindow()
width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

def grab_screen(X1, Y1, X2, Y2):
    desktop_dc = win32gui.GetWindowDC(hdesktop)
    img_dc = win32ui.CreateDCFromHandle(desktop_dc)
    mem_dc = img_dc.CreateCompatibleDC()
    screenshot = win32ui.CreateBitmap()
    screenshot.CreateCompatibleBitmap(img_dc, width, height)
    mem_dc.SelectObject(screenshot)
    mem_dc.BitBlt((X1, Y1), (X2-X1, Y2-Y1), img_dc, (X1, Y1), win32con.SRCCOPY)
    signedIntsArray = screenshot.GetBitmapBits(True)
    im_opencv = np.frombuffer(signedIntsArray, dtype = 'uint8')
    im_opencv.shape = (height, width, 4)
    im_opencv = im_opencv[Y1:Y2, X1:X2]
    mem_dc.DeleteDC()
    win32gui.DeleteObject(screenshot.GetHandle())
    return im_opencv