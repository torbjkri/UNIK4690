# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:29:03 2018

@author: wubst
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(1)
counter = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)

while(1):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,5), None)
        print(ret)
        if ret:
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            frame_copy = frame.copy()
            cv2.drawChessboardCorners(frame_copy, (8,5), corners2,ret)
            cv2.imshow('img',frame_copy)
            
            
            
        else:
            print('YALLA')
            cv2.imshow('img',frame)
            
        k = cv2.waitKey(10) & 0xFF
        if k == 32:
            cv2.imwrite('image{}.jpg'.format(counter), frame)
            counter += 1
        elif k == 27:
            print('Exit without problems')
            break
        
    else:
        print('Camera Fuckup')
        break
        

cap.release()
cv2.destroyAllWindows()