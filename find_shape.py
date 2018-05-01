# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:43:20 2018

@author: wubst
"""

import cv2
import numpy as np
import networkx as nx

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

def segment_contours(lines, max_distance = 100):
    contours = []
    success = 0
    for ii in range(0,lines.shape[0]):
        if len(contours) > 0:
            for sgmt in range(0,len(contours)):
                sgmtMean = np.mean(contours[sgmt], axis = 0)
                print(lines[ii,:,:].shape)
                if np.linalg.norm(lines[ii,:,:] - sgmtMean) < max_distance:
                    contours[sgmt] = np.vstack((contours[sgmt], lines[ii,:,:]))
                    success = 1
            if success == 0:
                contours.append(lines[ii])
        else:
            contours.append(lines[ii])
        success = 0
                
    return contours
            
    
    
    
    

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        
    return np.array([Px,Py])

def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers

cap = cv2.VideoCapture(0)

while(1):
    ret, img = cap.read()
    
    if ret:
        #img = cv2.resize(img, None, fx = .5, fy = .5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3,3), dtype = np.uint8))
        
        cv2.imshow('Dilated', dilated)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
        
cap.release()

rgb_planes = cv2.split(img)

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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
dilated = cv2.dilate(edges, np.ones((3,3), dtype = np.uint8))

lines = cv2.HoughLinesP(dilated, rho=1,theta=np.pi/180, threshold = 100, maxLineGap = 50, minLineLength = 50)


houghImg = img.copy()
for x1,y1,x2,y2 in lines[:,0,:]:

    cv2.line(houghImg, (x1,y1),(x2,y2),(0,255,0), 3 )
    
cv2.imshow('lines',houghImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours = segment_contours(lines, 30)

contours = [ii for ii in contours if ii.shape[0] > 1]


corner_image = img.copy()

for ii in contours:
    mean = np.uint32(np.mean(ii, axis = 0))
    x1, y1, x2, y2 = mean
    color = np.random.randint(0,255,3).tolist()
    
    cv2.circle(corner_image, (x1,y1), radius = 2, color = color, thickness = -1)
    cv2.circle(corner_image, (x2,y2), radius = 2, color = color, thickness = -1)
    
cv2.imshow('Intersections', corner_image)
cv2.waitKey(0)

mean = [np.mean(ii,axis = 0) for ii in contours]

connections = np.zeros((len(mean),len(mean)))

G = nx.Graph()

G.add_nodes_from(range(0,len(mean)))

for line1 in range(0,len(mean)):
    for line2 in range(line1,len(mean)):
        if line1 != line2:
            intersection = find_intersection(mean[line1],mean[line2])
            points = np.array([[mean[line1][0:2]],[mean[line1][2:4]],[mean[line2][0:2]],[mean[line2][2:4]]])
            for ii in range(0,2):
                for jj in range(2,4):
                    if ii != jj:
#                        print(np.linalg.norm(intersection - points[ii]))
                        if (np.linalg.norm(intersection - points[ii]) <= np.float64(20)) & \
                        (np.linalg.norm(intersection - points[jj]) <= np.float64(20)):
                            if ii == 0:
                                mean[line1][0:2] = intersection
                            elif ii == 1:
                                mean[line1][2:4] = intersection
                            if jj == 2:
                                mean[line2][0:2] = intersection
                            elif jj == 3:
                                mean[line2][2:4] = intersection
                                    
                            G.add_edge(line1,line2)
                            

H = G.to_directed()

cycles = nx.simple_cycles(H)
cycles = list(cycles)

for cycle in cycles:
    if len(cycle) ==4:
        print('HEI')
        for point in cycle:
            cv2.line(img, (np.uint16(mean[point][0]),np.uint16(mean[point][1])),(np.uint16(mean[point][2]),np.uint16(mean[point][3])), (255,0,0), 2 )
            
            
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



