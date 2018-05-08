# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:02:33 2018

@author: wubst
"""

import cv2
import numpy as np
import networkx as nx
import segmentation_functions as sf
import camera_parameters as cp
import itertools

original = cv2.imread('Figures/Screenshot.jpg')

target_scene = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)


K, dist = cp.camparam()

#target_scene = cv2.undistort(target_scene, K, dst = None, distCoeffs = dist)

#target_scene = sf.imageFixing(target_scene)


#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

#target_scene = clahe.apply(target_scene)
#target_scene = sf.imageFix2(target_scene)

cv2.imshow('original',target_scene)
kernel = np.ones((3,3), np.uint8)
edges = cv2.Canny(target_scene, threshold1 = 20, threshold2 = 150)
cv2.imshow('canny 1', edges)
#cv2.imshow('Canny', edges)
edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations = 1)
edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, kernel, iterations = 1)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 1)
#edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel = np.ones((3,3),np.uint8), iterations = 1)

cv2.imshow('closed', edges)
#cv2.imshow('Dilate+Erode', edges)

ret, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


contours = [ii for ii in contours if  len(ii) > 30 and cv2.isContourConvex(ii) == False ]



for ii, cnt in enumerate(contours):
    blank_image = np.zeros(target_scene.shape, np.uint8)
    cv2.drawContours(blank_image, cnt, -1, 255, 2)
    cv2.imshow('black' + str(ii), blank_image)
    lines = cv2.HoughLinesP(blank_image, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)
    if lines is not None:
        segments = sf.groupLineSegments(lines, thresholdTheta = 10.0*np.pi/180.0, thresholdRadius = 20.0)
        tmp = original.copy()
#        for kk in segments:
#            radius, theta, x1,y1,x2,y2 = kk
#            cv2.line(tmp, (np.uint32(x1),np.uint32(y1)),(np.uint32(x2),np.uint32(y2)), (0,255,0), 3)
        
        
    
    
        G = nx.Graph()
        
        G.add_nodes_from(range(0,len(segments)))
        
        G, segments = sf.createGraph(G, segments)
        
        H = G.to_directed()
        
        cycles = nx.simple_cycles(H)
        cycles = list(cycles)
        cycles = [sorted(ii) for ii in cycles if len(ii) == 4]
        cycles = sorted(cycles)
        cycles = list(cycles for cycles,_ in itertools.groupby(cycles))
        
        
        for cycle in cycles:
            if len(cycle) ==4:
                print('HEI')
                for point in cycle:
                    cv2.line(tmp, (np.uint16(segments[point][2]),np.uint16(segments[point][3])),(np.uint16(segments[point][4]),np.uint16(segments[point][5])), (255,0,0), 2 )
            
        cv2.imshow(str(ii), tmp)
        
cv2.imshow('result',target_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()    
    

#for ii, cnt in enumerate(contours):
#    black = np.zeros_like(target_scene)
#    cv2.drawContours(black, cnt, -1, 255,3)
#    dst = cv2.cornerHarris(black,2,3,0.04)
#    dst = cv2.dilate(dst,None)
#    tmp = original.copy()
#    tmp[dst>0.01*dst.max()]=[0,0,255]
#    cv2.imshow('black' + str(ii), black)
#    cv2.imshow(str(ii),tmp)
#    

#tmp = []

#for ii, cnt in enumerate(contours):
#    epsilon = 0.05*cv2.arcLength(cnt,True)
#    approx = cv2.approxPolyDP(cnt,epsilon,True)
#    tmp.append(approx)
#    if approx.shape[0] > 3:
#        cntimage = original.copy()
#        cv2.drawContours(cntimage, approx, -1, 0,3)
#        cv2.imshow(str(ii),cntimage)






cv2.waitKey(0)
cv2.destroyAllWindows()