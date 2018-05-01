# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:36:58 2018

@author: Torbjørn
"""

import cv2
import numpy as np

def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta:
                v_lines.append(line)
            elif abs(y2-y1) < delta:
                h_lines.append(line)
                
    return h_lines, v_lines

def segment_contours(lines, max_distance = 20):
    contours = []
    for ii in lines:
        if len(contours > 0):
            for sgmt in range(0,len(contours)):
                sgmtMean = np.mean(contours[sgmt], axis = 0)
                if np.linalg.norm(lines[ii,:,:] - sgmtMean) < max_distance:
                    contours[sgmt] = np.vstack(contours[sgmt], lines[ii])
                else:
                    contours.append(lines[ii])
        else:
            contours.append(lines[ii])
                
    return contours
            
    
    
    
    

def find_intersection(line1, line2):
    
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        
    return Px, Py

def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers

cap = cv2.VideoCapture(1)

while(1):
    ret, img = cap.read()
    
    if ret:
        img = cv2.resize(img, None, fx = .5, fy = .5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3,3), dtype = np.uint8))
        
        cv2.imshow('Dilated', dilated)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
        
cap.release()
lines = cv2.HoughLinesP(dilated, rho=1,theta=np.pi/180, threshold = 150, maxLineGap = 20, minLineLength = 100)
print(type(lines))
print(lines.shape)

delta = 30
h_lines, v_lines = segment_lines(lines, delta)

houghimg = img.copy()
for line in h_lines:
    for x1, y1, x2, y2 in line:
        color = [0,0,255]
        cv2.line(houghimg, (x1,y1), (x2,y2), color = color, thickness = 1)
        
for line in v_lines:
    for x1, y1, x2, y2 in line:
        color = [255,0,0]
        cv2.line(houghimg, (x1,y1), (x2,y2), color = color, thickness = 1)
        
cv2.imshow('Segmented Hough Lines', houghimg)
cv2.waitKey(0)

Px = []
Py = []

for h_line in h_lines:
    for v_line in v_lines:
        px, py = find_intersection(h_line, v_line)
        Px.append(px)
        Py.append(py)
        
intersectsimg = img.copy()
for cx, cy in zip(Px, Py):
    cx = np.round(cx).astype(int)
    cy = np.round(cy).astype(int)
    color = np.random.randint(0,255,3).tolist()
    cv2.circle(intersectsimg, (cx, cy), radius = 2, color = color, thickness = -1)
    
cv2.imshow('Intersections', intersectsimg)
cv2.waitKey(0)

P = np.float32(np.column_stack((Px, Py)))
nclusters = 4
centers = cluster_points(P, nclusters)
print(centers)

for cx, cy in centers:
    cx = np.round(cx).astype(int)
    cy = np.round(cy).astype(int)
    cv2.circle(img, (cx, cy), radius = 4, color = [0,0,255], thickness = -1)
    
cv2.imshow('Center for intersection clusters',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

                
                    