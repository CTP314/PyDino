import numpy
import cv2
from fastsc import grab_screen
import time
import glob
import win32api
import win32con
import pyautogui

X1, Y1, X2, Y2 = 0, 140, 960, 390
cactuses = glob.glob("C:\OI\TF\pydino\cactus\*.png")
dinoes = glob.glob("C:\OI\TF\pydino\dino\*.png")
birds = glob.glob("C:\OI\TF\pydino\\bird\*.png")

def trans(img, aver):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thre1 = 127 if aver <= 127 else 175
    ret, img = cv2.threshold(img, thre1, 255, cv2.THRESH_BINARY)
    img = img if aver <= 127 else 255 - img
    return img

def process_screen(original_image):
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    aver = numpy.mean(img)
    img = trans(original_image, aver)
    x1, x2, y1 = 0, X2, Y2 

    for dino in dinoes:
        c = trans(cv2.imread(dino), aver)
        w, h = c.shape[::-1]
        res = cv2.matchTemplate(img, c, cv2.TM_CCORR_NORMED)
        threshold = 0.9
        loc = numpy.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            x1 = max(x1, pt[0])
            y1 = min(y1, pt[1])
            cv2.rectangle(original_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    for cactus in cactuses:
        c = trans(cv2.imread(cactus), aver)
        w, h = c.shape[::-1]
        res = cv2.matchTemplate(img, c, cv2.TM_CCORR_NORMED)
        threshold = 0.9
        loc = numpy.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            if(pt[0] >= x1 and pt[1] >= y1):
                x2 = min(x2, pt[0])
            cv2.rectangle(original_image, pt, (pt[0] + w, pt[1] + h), (255, 255, 0), 2)

    for bird in birds:
        c = trans(cv2.imread(bird), aver)
        w, h = c.shape[::-1]
        res = cv2.matchTemplate(img, c, cv2.TM_CCORR_NORMED)
        threshold = 0.7
        loc = numpy.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            if(pt[0] >= x1 and pt[1] >= y1-30):
                x2 = min(x2, pt[0]-10)
            cv2.rectangle(original_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    # return img, x1, x2
    return original_image, x1, x2

if __name__ == '__main__' : 
    last_time = time.time()
    while(True):
        sc, a, b = process_screen(grab_screen(X1, Y1, X2, Y2))
        delta = time.time()-last_time
        print("delta : {} s, fps : {}".format(delta, 1/delta)) 
        last_time = time.time()
        cv2.imshow('windows', numpy.array(sc))
        if(win32api.GetAsyncKeyState(38) == -32767):
            print("jump %d %d %d"%(a, b, b-a))
        print(win32api.GetAsyncKeyState(38))
        print(b-a)
        if(b-a <= 200):
            print("jump")
            # win32api.keybd_event(40, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(38, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
            time.sleep(0.01)
            win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break