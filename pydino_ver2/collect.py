from numpy.lib.twodim_base import mask_indices
import fastsc
import win32api
import cv2
import time
import os

G = fastsc.Grab(0, 140, 960, 400)

def get_key_press(key):
    """Check if a key is being pressed.
    Needs changing for something that detects keypresses in applications.
    Returns:
        True/False if the selected key has been pressed or not.
    """
    return win32api.GetKeyState(key) < 0 

if __name__ == '__main__':
    last_time = time.time()
    os.makedirs("validation/1", exist_ok=True)
    os.makedirs("validation/0", exist_ok=True)
    st = [0, 0]
    while(True):
        s = G.grab_screen()
        cv2.imshow('windows', s)
        if get_key_press(key=38):
            # print("[action] jump")
            cv2.imwrite(f"validation/1/{st[1]}.png", s)
            time.sleep(0.1)
            st[1] += 1
        else:
            cur_time = time.time()
            if cur_time - last_time >= 0.1:
                last_time = cur_time
                cv2.imwrite(f"validation/0/{st[0]}.png", s)
                st[0] += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
