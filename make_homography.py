# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 11:09:04 2018

@author: wubst
"""

import cv2
import numpy as np


## Find good matches between images, and return them
def findGoodMatches(des1, des2, matchDetector):
    matches = matchDetector.knnMatch(des1, des2, k = 2)
    goodMatches = []
    
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)
            
    return goodMatches


##Given keypoints in two images and good matches between them, compute
## homography using RANSAC
def findHomography(kp1, kp2, goodMatches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)
    
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return homography