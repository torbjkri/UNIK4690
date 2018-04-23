# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:55:46 2018

@author: Torbjørn
"""

import numpy as np
import cv2
import camera as cm
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 20

cap = cv2.VideoCapture(1)

img1 = cv2.imread('QR_real.png.',0)
#screenshot = cm.screenshot(cap)

#img2 = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)



sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)


FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

img3 = None

true = 0

while(1):
    
    ret, frame = cap.read()
    
    print(ret)
    
    if ret:
        good = []
        img2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        kp2, des2 = sift.detectAndCompute(img2,None)
        matches = flann.knnMatch(des1,des2,k = 2)
        
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
                
            
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            
            h, w = img1.shape
            pts = np.float32([ [0,0],[0,h-1], [w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            
            img2 = cv2.polylines(img2,[np.int32(dst)], True,255,3,cv2.LINE_AA)
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[::10],None,**draw_params)
        
            
        else:
            print('NOt enough matches found - %d/%d',(len(good), MIN_MATCH_COUNT))
            matchesMask = None
            
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = None,
                           matchesMask = matchesMask,
                           flags = 2)
        
        true = 1
        
    if true == 1:
        cv2.imshow('result',img3)
        
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()


#
#img4 = cv2.warpAffine(img1,M,(h,w))
#
#plt.imshow(img4,'gray'),plt.show()

