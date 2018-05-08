# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:02:33 2018

@author: wubst
"""

import cv2
import numpy as np
import networkx as nx
import segmentation_functions as sf
import projection as pr
import math

import camera_parameters as cp
import itertools

def filterImage(image, kernel1 = np.ones((3,3), np.uint8), kernel2 = (3,3)):
    image = cv2.GaussianBlur(image, kernel2, 3)
    image = cv2.Canny(image, threshold1=20, threshold2=150)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel1, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1, iterations = 1)
    return image

def findContourPoly(image,cnt):
    for e in range(1,20):
        epsilon = 0.005 * e * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) <= 8:
            blank_image = np.zeros(image.shape, np.uint8)
            cv2.drawContours(blank_image, [approx], -1, 255, 2)
            lineSegments = cv2.HoughLinesP(blank_image, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50,maxLineGap=10000)
            if lineSegments is not None:
                lines = sf.groupLineSegments(lineSegments, thresholdTheta=10.0 * np.pi / 180.0, thresholdRadius=20.0)
                if 3 < len(lines) < 6:
                    for line in lines:
                        cv2.line(image, (np.uint16(line[2]), np.uint16(line[3])),
                             (np.uint16(line[4]), np.uint16(line[5])), (255, 0, 0), 2)
            break
    return image

def findContourNX(image, cnt):
    intersections = []
    blank_image = np.zeros(image.shape, np.uint8)
    cv2.drawContours(blank_image, cnt, -1, 255, 2)

    # Compute Houghlines of contour
    lineSegments = cv2.HoughLinesP(blank_image, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

    if lineSegments is not None:
        # Combine line segments with equal theta and radius into a single line
        lines = sf.groupLineSegments(lineSegments, thresholdTheta = 10.0*np.pi/180.0, thresholdRadius = 20.0)

        # Create graph object and do some magic
        G = nx.Graph()
        G.add_nodes_from(range(0,len(lines)))
        G, lines = sf.createGraph(G, lines)
        H = G.to_directed()
        cycles = list(nx.simple_cycles(H))
        cycles = [ii for ii in cycles if len(ii) == 4]
        if len(cycles) != 0 and len(lines) == 4:
            for point in cycles[0]:
                cv2.line(image, (np.uint16(lines[point][2]),np.uint16(lines[point][3])),(np.uint16(lines[point][4]),np.uint16(lines[point][5])), (255,0,0), 2 )
            pnt1 = sf.find_intersection(lines[cycles[0][3]],lines[cycles[0][0]])
            pnt2 = sf.find_intersection(lines[cycles[0][0]], lines[cycles[0][1]])
            pnt3 = sf.find_intersection(lines[cycles[0][1]], lines[cycles[0][2]])
            pnt4 = sf.find_intersection(lines[cycles[0][2]], lines[cycles[0][3]])
            Spnt1 = math.sqrt(pnt1[0]**2+pnt1[1]**2)
            Spnt2 = math.sqrt(pnt2[0]**2+pnt2[1]**2)
            Spnt3 = math.sqrt(pnt3[0]**2+pnt3[1]**2)
            Spnt4 = math.sqrt(pnt4[0]**2+pnt4[1]**2)
            k = 0
            if min(Spnt1,Spnt2,Spnt3,Spnt4) == Spnt1:
                intersections = [pnt1, pnt2, pnt3, pnt4]
                k=1
            elif min(Spnt1, Spnt2, Spnt3, Spnt4) == Spnt2:
                intersections = [pnt2, pnt3, pnt4, pnt1]
                k=2
            elif min(Spnt1,Spnt2,Spnt3,Spnt4) == Spnt3:
                intersections = [pnt3, pnt4, pnt1, pnt2]
                k=3
            elif min(Spnt1,Spnt2,Spnt3,Spnt4) == Spnt4:
                intersections = [pnt4, pnt1, pnt2, pnt3]
                k=4
            middle1 = intersections[1]
            middle2 = intersections[3]
            if middle1[1] > middle2[1]:
                print('intersections_prev:', intersections,k)
                intersections[1] = middle2
                intersections[3] = middle1
                print('intersections:', intersections)
                print('asdf')
    return image, np.array(intersections)

def findContourPoints(imageName = 'Figures/corridor.jpg'):
    # 1. Read image and convert to grayscale
    original = cv2.imread(imageName)
    target_scene = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
    scene_with_projection = target_scene.copy()

    # 2. Filter grayscale and obtain list of contours
    edges = filterImage(target_scene)
    ret, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    temp = []
    outer_contour = 0
    child = hierarchy[0,0,2]
    while(outer_contour >=0):
        if child >= 0:
            temp.append(contours[child])
        else:
            temp.append(contours[outer_contour])

        outer_contour = hierarchy[0,outer_contour,0]
        child = hierarchy[0,outer_contour,2]
    #contours = temp



    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
    # 3. Loop through contours
    target_scene2 = target_scene.copy()
    for ii, cnt in enumerate(contours):
        target_scene, pts_dst = findContourNX(target_scene,cnt)
        if len(pts_dst) == 4:
            target_projection = cv2.imread('Figures/banksy.jpg', 0)
            height, width = target_projection.shape
            pnt1 = [0, 0]
            pnt2 = [width, 0]
            pnt3 = [width, height]
            pnt4 = [0, height]
            pts_src = np.array([pnt1,pnt2,pnt3,pnt4])
            print(pts_src)
            print(np.uint16(pts_dst))
            homography, status = cv2.findHomography(pts_src, pts_dst)
            scene_with_projection = pr.projectImage(target_scene, target_projection, homography)
            cv2.imshow('scene with projection',scene_with_projection)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #target_scene2 = findContourPoly(target_scene2, cnt)
    #cv2.imshow('NX',target_scene)
    #cv2.imshow('scene with projection',scene_with_projection)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return scene_with_projection

imageList = ['corridor',
             'camshothome0',
             'camshothome1',
             'camshothome2',
             'camshothome3',
             'Screenshot',
             'Screenshot2',
             'Screenshot3',
             ]

imageList = [
             'camshothome0',
             ]


for name in imageList:
    image_result = findContourPoints('Figures/'+name+'.jpg')
    cv2.imshow(name,image_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
