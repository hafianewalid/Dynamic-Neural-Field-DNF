import cv2
import numpy as np

import animation
import DNF

'''
Pour installer opencv:
sudo apt-get install opencv* python3-opencv


Si vous avez des problèmes de performances, vous pouvez calculer les convolutions plus rapidement avec :

lateral = signal.fftconvolve(activation, kernel, mode='same')

/!\ Votre kernel doit être de taille impaire pour que la convolution fonctionne correctement (taille_dnf * 2 - 1 par exemple).
'''

images_path = "pacman/"
image_size = (380, 455)
window_pos = (50, 280)
window_size = (150, 150)

# Pour Opencv : Hue [0,179], Saturation [0,255], Value [0,255]
def selectByColor(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',hsv)
    #key = cv2.waitKey(0)


    low_yellow = np.array([0,140,100])
    high_yellow = np.array([60,255,255])
    yellow_mask = cv2.inRange(frame,low_yellow,high_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    #cv2.imshow('yell',yellow)
    gray = cv2.cvtColor(yellow,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('yell',gray)
    #key = cv2.waitKey(0)


    frame=gray/np.max(gray)
    #cv2.imshow('yell_norm',gray)
    #key = cv2.waitKey(0)

    return frame


def findCenter(potentials):

    ind=np.argwhere(potentials>0.1)
    centre=np.sum(ind,axis=0)
    if ind.shape[0]>0:
        centre=(centre[1]/ind.shape[0],centre[0]/ind.shape[0])
    return centre
    #pass


def moveWindow(center, speed):
    global window_pos

    xt=(center[0]-window_size[0]/2)
    yt=(center[1]-window_size[1]/2)
    dx=int((xt-window_pos[0])*speed)
    dy=int((yt-window_pos[1])*speed)
    if(0<window_pos[0]+dx<image_size[0]and 0<window_pos[1]+dy<image_size[1]):
        window_pos=(window_pos[0]+dx,window_pos[1]+dy)
    #pass

Speed=1
NB_it=1
def track(frame):
    input = selectByColor(frame)
    #dnf.Input_Map = input
    dnf.set_input(input)

    for i in range(NB_it):
     dnf.syncronous_run()

    cv2.imshow("Input_Map", input)
    cv2.imshow("Potentials", dnf.Map)
    center = findCenter(dnf.Map)
    #cv2.circle(dnf.Input_Map,(int(center[0]),int(center[1])),1,(255, 0, 0),2)

    moveWindow(center,Speed)

def speed_trackbar(val):
    global Speed
    Speed=val/100
def NBit_trackbar(val):
    global NB_it
    NB_it=val

if __name__ == '__main__':
    frame = cv2.imread(images_path+"pacman00001.png")
    cv2.imshow("Frame", frame)
    cv2.createTrackbar("Speed","Frame",100,100,speed_trackbar)
    cv2.createTrackbar("Nbit","Frame",1,5,NBit_trackbar)

    dnf = DNF.DNF(size=(image_size[1],image_size[0]))

    for i in range(1,196):
        frame = cv2.imread(images_path + "pacman{0:05d}.png".format(i))
        track(frame)
        frame_np = np.asarray(frame)
        window = frame_np[window_pos[1]:window_pos[1]+window_size[1], window_pos[0]:window_pos[0]+window_size[0]]
        cv2.imshow("Input", window)

        cv2.rectangle(frame,window_pos,(window_pos[0]+window_size[0],window_pos[1]+window_size[1]),color=(0, 255, 0), thickness=3)
        cv2.imshow("Frame",frame)

        key = cv2.waitKey(5)
        if key == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
