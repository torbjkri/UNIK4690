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

def filterImage(image, kernel1 = np.ones((3,3), np.uint8), kernel2 = tuple(3,3)):
    image = cv2.GaussianBlur(image, kernel2, 3)
    image = cv2.Canny(image, threshold1=20, threshold2=150)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel1, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1, iterations = 1)
    return image

    return image
def findContours(imageName = 'Figures/corridor.jpg'):
    kernel = np.ones((3,3), np.uint8)
    original = cv2.imread(imageName)
    # Get grey scale image
    target_scene = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
    edges = filterImage(target_scene, kernel1=np.ones((3,3), np.uint8))
    #cv2.imshow('edges',edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # Retrieve external contours and filter by area
    ret, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]

    # Loop through contours
    for ii, cnt in enumerate(contours):
        # Project contour on black image
        blank_image = np.zeros(target_scene.shape, np.uint8)
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

                #cv2.imshow(str(len(lines))+' '+str(len(lineSegments))+' '+str(len(cycles)), blank_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                for point in cycles[0]:
                    cv2.line(target_scene, (np.uint16(lines[point][2]),np.uint16(lines[point][3])),(np.uint16(lines[point][4]),np.uint16(lines[point][5])), (255,0,0), 2 )
    return target_scene
imageList = ['corridor',
             'frames',
             'camshothome0',
             'camshothome1',
             'camshothome2',
             'camshothome3',
             'Screenshot',
             'Screenshot2',
             'Screenshot3',
             ]

for name in imageList:
    cv2.imshow(name,findContours('Figures/'+name+'.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()
