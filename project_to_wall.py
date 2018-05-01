# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:55:18 2018

@author: wubst
"""

import numpy as np
import cv2
import camera as cm
import projection as pr
from matplotlib import pyplot as plt


# Input images
QR_template = cv2.imread('Figures/QR_real.png',0)
target_projection = cv2.imread('Figures/banksy.jpg',0)
MIN_MATCH_COUNT = 20
input_scene = 1
homography_computation_method = 2


# 1. Choose input_scene type (1 = Figures/Screenshot.jpg, 2 = take sceenshot)
if input_scene == 1:
    target_scene = cv2.imread('Figures/Screenshot.jpg', 0)
elif input_scene == 2:
    cap = cv2.VideoCapture(1)
    screenshot = cm.screenshot(cap)
    target_scene = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    cap.release()

# 2. Choose homography_computation_method (1 = QR_code, 2 = detect squares in scene)
if homography_computation_method == 1:
    # 2.1 Get lists of keypoints (kp_) and descriptors (des_) of template and target scene
    sift = cv2.xfeatures2d.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(QR_template,None)
    kp_target, des_target = sift.detectAndCompute(target_scene,None)

    # Get descriptor matches using FLANN (defined in function)
    goodMatches = pr.findGoodMatches(des_template, des_target)

    # If enough matches, compute homography and print projection on scene
    if len(goodMatches) > MIN_MATCH_COUNT:
        homography = pr.findHomography(kp_template, kp_target, goodMatches)
        scene_with_projection = pr.projectImage(target_scene, target_projection, homography)
    else:
        print('NOt enough matches found - %d/%d', (len(goodMatches), MIN_MATCH_COUNT))

elif homography_computation_method == 2:
    # 2.2 Detect squares in scene and compute homography
    img = target_scene

    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    # Canny edge detection
    edges = cv2.Canny(target_scene, 20, 200)

    #edges = cv2.GaussianBlur(edges, (5,5), 1)
    #edges = cv2.dilate(edges, np.ones((5,5), dtype = np.uint8))

    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
    N = lines.shape[0]

    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('result',img)
    cv2.waitKey(0)

    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[4]
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    cv2.imshow('Contours', img)

    cnt = contours[1]
# 3. Show result
#cv2.imshow('projected',scene_with_projection)
#cv2.waitKey(0)
cv2.destroyAllWindows()
