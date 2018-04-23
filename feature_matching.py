# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:23:28 2018

@author: wubst
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import camera as cm


cm.screenshot()

#img1 = cv2.imread('QR_real.png',0)
img1 = cv2.imread('screenshot.jpg',0)

orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create(3000)

kp1orb, des1orb = orb.detectAndCompute(img1,None)
kp1sift , des1sift = sift.detectAndCompute(img1,None)

bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
bf2 = cv2.BFMatcher()

cap = cv2.VideoCapture(0)

cv2.namedWindow('sift result',cv2.WINDOW_NORMAL)
cv2.namedWindow('ORB result',cv2.WINDOW_NORMAL)

while(cap.isOpened == False):
    print('Opening Camera')

while(1):
    ret, frame = cap.read()
    
    if ret:
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2orb, des2orb = orb.detectAndCompute(img2,None)
        kp2sift, des2sift = sift.detectAndCompute(img2,None)    
        matchesorb = bf1.match(des1orb,des2orb)
        matchessift = bf2.knnMatch(des1sift,des2sift, k = 5)
        
        
        if matchesorb:
            matchesorb = sorted(matchesorb, key = lambda x:x.distance)
            #matchessift = sorted(matchessift, key = lambda x:x.distance)
            good = []
            for m, n in matchessift:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            
            matchessift = sorted(good, key = lambda x:x.distance)
            img3sift = img2.copy()
            img3orb = img2.copy()
            
            img3orb = cv2.drawMatches(img1,kp1orb,img2,kp2orb,matchesorb[:10], outImg = img3orb, flags = 2)
            img3sift = cv2.drawMatches(img1,kp1sift,img2,kp2sift,good[:10], outImg = img3sift, flags = 2)
            
            cv2.imshow('sift result', img3sift)
            cv2.imshow('ORB result', img3orb)
        
        #cv2.imshow('original',img2)
        
        
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:
            break
        
cap.release()
cv2.destroyAllWindows()