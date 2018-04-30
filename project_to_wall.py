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

img1 = cv2.imread('Figures/QR_real.png',0)

screenshot = cm.screenshot(cap)

img2 = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

img4 = cv2.imread('Figures/banksy.jpg',0)
#screenshot = cm.screenshot(cap)

#img2 = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image')
cv2.setMouseCallback('image',placeImage)

while(1):
    cv2.imshow('image', img2)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

img3 = None

true = 0



    
#ret, frame = cap.read()



good = []
#img2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


matches = flann.knnMatch(des1,des2,k = 2)

for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
    
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    
    mean = np.mean(dst_pts, axis = 0)
    
    difference = (mouseX,mouseY) - mean
    
    dst_pts = dst_pts + difference
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    h, w = img4.shape
    x0 = 0
    y0 = 0
    pts = np.float32([ [x0,y0],[x0,y0+h-1], [x0+w-1,y0+h-1],[x0+w-1,y0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    img2 = cv2.polylines(img2,[np.int32(dst)], True,255,3,cv2.LINE_AA)
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)


    img2_bg = pr.projectImage(img2, img4, M)
    #img5_tot = cv2.bitwise_and(img2_bg, img5)
#    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
    

    
else:
    print('NOt enough matches found - %d/%d',(len(good), MIN_MATCH_COUNT))
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

