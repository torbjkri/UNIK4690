# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:55:18 2018

@author: wubst
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:55:46 2018

@author: Torbjørn
"""

import numpy as np
import cv2
import camera as cm
import projection as pr
from matplotlib import pyplot as plt


def placeImage(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x,y


MIN_MATCH_COUNT = 20

cap = cv2.VideoCapture(1)

QR_template = cv2.imread('Figures/QR_real.png',0)

screenshot = cm.screenshot(cap)

target_scene = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

img4 = cv2.imread('Figures/banksy.jpg',0)
#screenshot = cm.screenshot(cap)

#img2 = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image')
cv2.setMouseCallback('image',placeImage)

while(1):
    cv2.imshow('image', target_scene)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)

sift = cv2.xfeatures2d.SIFT_create()

kp_template, des_template = sift.detectAndCompute(QR_template,None)
kp_target, des_target = sift.detectAndCompute(target_scene,None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

goodMatches = pr.findGoodMatches(des_template, des_target, flann)
        
    
if len(goodMatches) > MIN_MATCH_COUNT:
    M = pr.findHomography(kp_template, kp_target, goodMatches)

    img2_bg = pr.projectImage(target_scene, img4, M)
    
else:
    print('NOt enough matches found - %d/%d',(len(goodMatches), MIN_MATCH_COUNT))
    matchesMask = None
    

true = 1

cap.release()
cv2.imshow('projected',img2_bg)
#cv2.imshow('result',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

    

cv2.destroyAllWindows()


#
#img4 = cv2.warpAffine(img1,M,(h,w))
#
#plt.imshow(img4,'gray'),plt.show()

