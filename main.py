# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:52:46 2018

@author: wubst
"""


import cv2
import numpy as np

img = cv2.imread('screenshot.jpg')

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

kp = orb.detect(img, None)

kp, des = orb.compute(img, kp)

img2 = cv2.drawKeypoints(img, kp,img , color = (0,255,0), flags = 0)

cv2.imshow('keypoints',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()