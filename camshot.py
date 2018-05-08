
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:01:37 2018

@author: wubst
"""

import cv2

cap = cv2.VideoCapture(1)
    
cv2.namedWindow('stream',cv2.WINDOW_NORMAL)
cv2.namedWindow('screenshot',cv2.WINDOW_NORMAL)

screenshot = None
true = 0
counter = 0

while(1):
    
    ret, frame = cap.read()
    
    if ret:
        
        cv2.imshow('stream',frame)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k == 32:
            screenshot = frame.copy()
            cv2.imwrite('Figures/camshothome{}.jpg'.format(str(counter)), screenshot)
            counter += 1
            true = 1
            
        elif k == 27:

            break
                
        if true == 1:
            cv2.imshow('screenshot', screenshot)
            
cap.release()
            
cv2.destroyAllWindows()
