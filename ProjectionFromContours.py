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


# Return filtered grey image
def filterImage(image, kernel1 = np.ones((3,3), np.uint8), kernel2 = (3,3)):
    image = cv2.GaussianBlur(image, kernel2, 3)
    image = cv2.Canny(image, threshold1=20, threshold2=150)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel1, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1, iterations = 2)
    return image

# Return image with single contour (NOT WORKING)
def findContourPoly(image,cnt):
    for e in range(1,20):
        epsilon = 0.005 * e * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) <= 4:
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

# Return array with four cornerpoints [x,y] of the contour
def findContourNX(image, cnt):
    cornerPoints = []
    blank_image = np.zeros(image.shape, np.uint8)
    cv2.drawContours(blank_image, cnt, -1, 255, 2)

    # Compute Houghlines of contour (x1,y1 x2,y2)
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
            #for point in cycles[0]:
            #    cv2.line(image, (np.uint16(lines[point][2]),np.uint16(lines[point][3])),(np.uint16(lines[point][4]),np.uint16(lines[point][5])), (255,0,0), 2 )
            # Take first cycle and compute corner points
            cycle = cycles[0]
            cornerPoints = []
            cornerPoints.append(sf.find_intersection(lines[cycle[0]], lines[cycle[1]]))
            cornerPoints.append(sf.find_intersection(lines[cycle[1]], lines[cycle[2]]))
            cornerPoints.append(sf.find_intersection(lines[cycle[2]], lines[cycle[3]]))
            cornerPoints.append(sf.find_intersection(lines[cycle[3]], lines[cycle[0]]))

            # Compute R distance to origin (to determine top right corner of homography)
            cornerRadius = []
            for pnt in cornerPoints:
                cornerRadius.append(math.sqrt(pnt[0]**2+pnt[1]**2))

            # Find cornerpoint of top left corner
            rSmall = min(cornerRadius)
            if rSmall == cornerRadius[1]:
                cornerPoints = [cornerPoints[1], cornerPoints[2], cornerPoints[3], cornerPoints[0]]
            elif rSmall == cornerRadius[2]:
                cornerPoints = [cornerPoints[2], cornerPoints[3], cornerPoints[0], cornerPoints[1]]
            elif rSmall == cornerRadius[3]:
                 cornerPoints = [cornerPoints[3], cornerPoints[0], cornerPoints[1], cornerPoints[2]]

        # Flip cornerpoints 1 and 3 if x1 > x3
            corner1 = cornerPoints[1]
            corner3 = cornerPoints[3]
            if corner1[1] > corner3[1]:
                cornerPoints[1] = corner3
                cornerPoints[3] = corner1
    return np.array(cornerPoints)

# Return filtered list of contours (no child, no len < threshold)
def filterContours(contours, hierarchy, Threshold_len = 400):
    # step 2.1: Remove all contours with length less than 400
    for ii, cnt in enumerate(contours):
        if len(cnt) < Threshold_len:
            contours[ii] = None
    # step 2.2: Remove all contours that still have a child contour
    for ii, cnt in enumerate(contours):
        if cnt is not None:
            cnt_child = hierarchy[0][ii][2]
            if cnt_child >= 0:
                if contours[cnt_child] is not None:
                    contours[ii] = None
            else:
                contours[ii] = None
    # step 2.3: Remove all None contours
    contours = [cnt for cnt in contours if cnt is not None]
    return contours

# Return gray image with projection
def projectImage(imageName = 'Figures/corridor.jpg', projectionName = 'Figures/banksy.jpg'):
    # 1. Read image and convert to grayscale
    original = cv2.imread(imageName)
    projection = cv2.imread(projectionName)
    target_scene = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
    target_projection = cv2.cvtColor(projection.copy(), cv2.COLOR_BGR2GRAY)
    scene_with_projection = target_scene.copy()

    # 2. Get corner points of target projection for homography calculation (step 5)
    height, width = target_projection.shape
    pts_src = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # 3. Filter grayscale and obtain list of contours
    edges = filterImage(target_scene)
    ret, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # 4. Remove/filter small and duplicate contours:
    contours = filterContours(contours, hierarchy, Threshold_len=400)

    # 5. Find homography per contours and project target_projection
    for ii, cnt in enumerate(contours):
        pts_dst = findContourNX(target_scene,cnt)
        if len(pts_dst) == 4:

            # Compute homography and project target_projection on top of image
            homography, status = cv2.findHomography(pts_src, pts_dst)
            scene_with_projection = pr.projectImage(scene_with_projection, target_projection, homography)
            #cv2.imshow('scene with projection',scene_with_projection)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    return scene_with_projection

#imageList = ['corridor',
#             'camshothome0',
#             'camshothome1',
#             'camshothome2',
#             'camshothome3',
#             'Screenshot',
#             'Screenshot2',
#             'Screenshot3',
#             ]

#for name in imageList:
#    image_result = projectImage('Figures/'+name+'.jpg')
#    cv2.imshow(name,image_result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
