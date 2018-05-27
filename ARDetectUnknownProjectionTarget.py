# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:33:53 2018

@author: wubst
"""

import cv2
import numpy as np
#import ProjectionFromContours as pfc
#import projection as pr
import networkx as nx
import math

class Image:
    def __init__(self,imageFile):
        self.image = cv2.imread(imageFile)

# Proect RGB projection image, and merge with scene
def projectImage(img1, img2, homography, thresh=15, maxval = 255):
    # Transform artwork according to the homography
    
    w1,h1 = img1.shape[:2]
    
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray += 20
    img2_warped = cv2.warpPerspective(img2, homography, dsize=(h1, w1))
    img2_gray_warped = cv2.warpPerspective(img2_gray,homography, dsize=(h1,w1))
    ret, mask = cv2.threshold(img2_gray_warped, thresh, maxval, cv2.THRESH_BINARY)  
    kernel = np.ones((20,20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    inv = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(img1,img1,mask = inv)
    img2_fg = cv2.bitwise_and(img2_warped,img2_warped, mask = mask)
    result = cv2.add(img1_bg, img2_fg)
    return result

# Preprocessing of scene, and detect edges, return binary edge image
def filterImage(image, kernel1 = np.ones((3,3), np.uint8), kernel2 = (3,3)):
    image = cv2.GaussianBlur(image, kernel2, 3)
    image = cv2.Canny(image, threshold1=20, threshold2=150)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel1, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1, iterations = 2)
    return image

# Filter contours, return list of remaining contours
def filterContours(contours, hierarchy, Threshold_len = 400):
    # Remove all contours with length less than 400
    for ii, cnt in enumerate(contours):
        if len(cnt) < Threshold_len:
            contours[ii] = None
    #Remove all contours that still have a child contour
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



# Return array with four cornerpoints [x,y] of the contour
def findContourNX(image, cnt):
    cornerPoints = []
    blank_image = np.zeros(image.shape, np.uint8)
    cv2.drawContours(blank_image, cnt, -1, 255, 2)

    # Compute Houghlines of contour (x1,y1 x2,y2)
    lineSegments = cv2.HoughLinesP(blank_image, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

    if lineSegments is not None:
        # Combine line segments with equal theta and radius into a single line
        lines = groupLineSegments(lineSegments, thresholdTheta = 10.0*np.pi/180.0, thresholdRadius = 20.0)

        # Create graph object and do some magic
        G = nx.Graph()
        G.add_nodes_from(range(0,len(lines)))
        G, lines = createGraph(G, lines)
        H = G.to_directed()
        cycles = list(nx.simple_cycles(H))
        cycles = [ii for ii in cycles if len(ii) == 4]
        if len(cycles) != 0 and len(lines) == 4:
            #for point in cycles[0]:
            #    cv2.line(image, (np.uint16(lines[point][2]),np.uint16(lines[point][3])),(np.uint16(lines[point][4]),np.uint16(lines[point][5])), (255,0,0), 2 )
            # Take first cycle and compute corner points
            cycle = cycles[0]
            cornerPoints = []
            cornerPoints.append(find_intersection(lines[cycle[0]], lines[cycle[1]]))
            cornerPoints.append(find_intersection(lines[cycle[1]], lines[cycle[2]]))
            cornerPoints.append(find_intersection(lines[cycle[2]], lines[cycle[3]]))
            cornerPoints.append(find_intersection(lines[cycle[3]], lines[cycle[0]]))

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

# Return intersection [x,y] of two lines given their endpoints
def find_intersection(line1, line2):
    _, _, x1, y1, x2, y2 = line1
    _, _, x3, y3, x4, y4 = line2
    
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/ \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        
    return np.array([Px,Py])


# Given lines given by their endpoints, group them together based on
# radius and theta
def groupLineSegments(lines, thresholdTheta,thresholdRadius):
    lineSegmentGroups = []
#    print('THRESHOLD')
#    print(thresholdTheta)
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

        # Group segments by radius and theta
        N = len(lineSegmentGroups)
        addNewLine = True
        if N > 0:
            for i in range(N):
                radius_SG = np.mean(lineSegmentGroups[i],axis = 0)[0]
                theta_SG = np.mean(lineSegmentGroups[i],axis = 0)[1]

                if abs(abs(radius) - abs(radius_SG)) < thresholdRadius and abs(abs(theta) - abs(theta_SG)) < thresholdTheta:
                    
                    lineSegmentGroups[i] = np.vstack((lineSegmentGroups[i], np.array([radius, theta,x1,y1,x2,y2])))       #.append([x1,y1,x2,y2])
                    addNewLine = False
                    break
            if addNewLine:
                lineSegmentGroups.append(np.array([radius, theta, x1, y1, x2, y2])[np.newaxis,:])
        else:
            lineSegmentGroups.append(np.array([radius, theta, x1, y1, x2, y2])[np.newaxis,:])

    # Merge each group of lines into one line
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

# Create undirected graph for edges
def createGraph(G, segments):
    for line1 in range(0,len(segments)):
        for line2 in range(line1,len(segments)):
            if line1 != line2:
                intersection = find_intersection(segments[line1],segments[line2])
                points = np.array([[segments[line1][2:4]],[segments[line1][4:6]],[segments[line2][2:4]],[segments[line2][4:6]]])
                for ii in range(0,2):
                    for jj in range(2,4):
                        if ii != jj:
                            if (np.linalg.norm(intersection - points[ii]) <= np.float64(20)) & \
                            (np.linalg.norm(intersection - points[jj]) <= np.float64(20)):
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




    
def mainFunction(imageFile, liveView = True):
    
    # Create list of projection images
    images = []
    for ii in range(0,len(imageFile)):
        images.append(Image(imageFile[ii]))
        
    # If camera is not used, use a screenshot to detect shapes
    if liveView == False:
        original = cv2.imread('Figures/camshokjeller0.jpg')
        target_scene = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
    else:
        cap = cv2.VideoCapture(0)
    
    counter = 0 #For keeping track of saved images 
    
    

    while(1):
        
        if liveView:
            ret, original = cap.read()
            if ret:
                target_scene = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
                
        if ret:
            
            # Filter image, detect edges, find and filter contours
            edges = filterImage(target_scene)
            ret, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            contours = filterContours(contours, hierarchy, Threshold_len=200)
            
            
            # Find homography per contours and project target_projection
            for ii, cnt in enumerate(contours):
                pts_dst = findContourNX(target_scene,cnt)
                if len(pts_dst) == 4:
                    target_projection = images[ii%len(imageFile)].image.copy()
                    height, width = target_projection.shape[:2]
                    pts_src = np.array([[0, 0], [width, 0], [width, height], [0, height]])
                    homography, status = cv2.findHomography(pts_src, pts_dst)
                    original = projectImage(original, target_projection, homography)
                    
            cv2.imshow('scene with projection',original)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
        if k == 32:
            cv2.imwrite('Figures/manyFrames{}.jpg'.format(counter),original)
            counter += 1

def main():
    imageFile = imageFile = ['Figures/banksy.jpg', 'Figures/gtalady2.jpg','Figures/fakelove.jpg']
    
    mainFunction(imageFile, True)
    
if __name__ == "__main__": main()