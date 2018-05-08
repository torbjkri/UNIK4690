# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:06:11 2018

@author: wubst
"""

import cv2
import numpy as np
import math

def groupLineSegments(lines, thresholdTheta,thresholdRadius):
    lineSegmentGroups = []
    for line in lines:
        x1,y1,x2,y2 = line[0]

        # compute theta (-0.5pi to 0.5 pi) and radius (negative and positive)
        if (x2 - x1) == 0:
            theta = math.pi/2.0
            radius = x1
        elif (y2-y1) == 0:
            theta = 0.0
            radius = y1
        else:
            gradient = (y2 - y1) / (x2 - x1)
            y_at_x0 = y1 - (x1 * gradient)
            x_at_y0 = x1 - (y1 / gradient)
            theta = math.atan2(gradient, 1)
            radius = 0.5 * (y_at_x0 * math.cos(theta) - x_at_y0 * math.sin(theta))
        #rx = radius * -math.sin(theta)
        #ry = radius * math.cos(theta)

        # Group segments by radius and theta
        N = len(lineSegmentGroups)
        addNewLine = True
        if N > 0:
            for i in range(N):
                radius_SG = np.mean(lineSegmentGroups[i],axis = 0)[0]
                theta_SG = np.mean(lineSegmentGroups[i],axis = 0)[1]

                if abs(abs(radius) - abs(radius_SG)) < thresholdRadius and abs(theta - theta_SG) < thresholdTheta:

                    lineSegmentGroups[i] = np.vstack((lineSegmentGroups[i], np.array([radius, theta,x1,y1,x2,y2])))       #.append([x1,y1,x2,y2])
                    addNewLine = False
                    break
            if addNewLine:
                lineSegmentGroups.append(np.array([radius, theta, x1, y1, x2, y2])[np.newaxis,:])
        else:
            lineSegmentGroups.append(np.array([radius, theta, x1, y1, x2, y2])[np.newaxis,:])
            
    result = []
    for segment in lineSegmentGroups:
        if np.mean(segment,axis = 0)[1] <= 0:
            x1 = np.amin(segment[:,[2,4]])
            x2 = np.amax(segment[:,[2,4]])
            y1 = np.amax(segment[:,[3,5]])
            y2 = np.amin(segment[:,[3,5]])
            theta = np.mean(segment[:,1])
            radius = np.mean(segment[:,0])
            result.append(np.array([radius, theta, x1,y1,x2,y2]))
        else:
            x1 = np.amin(segment[:,[2,4]])
            x2 = np.amax(segment[:,[2,4]])
            y1 = np.amin(segment[:,[3,5]])
            y2 = np.amax(segment[:,[3,5]])
            theta = np.mean(segment[:,1])
            radius = np.mean(segment[:,0])
            result.append(np.array([radius, theta, x1,y1,x2,y2]))
            
    return result


def imageFixing(target_scene):
    
    rgb_planes = cv2.split(target_scene)
    
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    
    target_scene = result_norm
    
    return target_scene

def imageFix2(img):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    close = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(img)/(close)
    res = np.uint8(cv2.normalize(div,div,0,150,cv2.NORM_MINMAX))
    
    return res

def findLines(lines, thresholdTheta = 4.0*math.pi/180.0, thresholdRadius = 10.0):
    linesFiltered=[]

    # Group lines by theta and radius
    linesGroups = groupLineSegments(lines, thresholdTheta, thresholdRadius)

    # Compute individual lines per group
    for group in linesGroups:
        newLines = (connectSegments(group[2]))
        for line in newLines:
            line.append(group[0])
            line.append(group[1])
            linesFiltered.append(line)

    return linesFiltered


def collectSegments(segments, radius):
    for ii, sgmt1 in enumerate(segments):
        for jj, sgmt2 in enumerate(segments):
            if sgmt1 != sgmt2:
                if np.linalg.norm(sgmt1[2:4] - sgmt2[2:4]) + \
                    np.linalg.norm(sgmt1[4:6] - sgmt2[4:6]) < 2*radius:
                        segments.remove(sgmt2)
                        segments[ii] = sgmt1[ii]
                    
                    

def createGraph(G, segments):
    for line1 in range(0,len(segments)):
        for line2 in range(line1,len(segments)):
            if line1 != line2:
                intersection = find_intersection(segments[line1],segments[line2])
                points = np.array([[segments[line1][2:4]],[segments[line1][4:6]],[segments[line2][2:4]],[segments[line2][4:6]]])
                for ii in range(0,2):
                    for jj in range(2,4):
                        if ii != jj:
    #                        print(np.linalg.norm(intersection - points[ii]))
                            if (np.linalg.norm(intersection - points[ii]) <= np.float64(20)) & \
                            (np.linalg.norm(intersection - points[jj]) <= np.float64(20)):
                                print(np.linalg.norm(intersection - points[ii]))
                                if ii == 0:
                                    segments[line1][2:4] = intersection
                                elif ii == 1:
                                    segments[line1][4:6] = intersection
                                if jj == 2:
                                    segments[line2][2:4] = intersection
                                elif jj == 3:
                                    segments[line2][4:6] = intersection
                                        
                                G.add_edge(line1,line2)
    return G, segments

def find_intersection(line1, line2):
    _, _, x1, y1, x2, y2 = line1
    _, _, x3, y3, x4, y4 = line2
    
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        
    return np.array([Px,Py])