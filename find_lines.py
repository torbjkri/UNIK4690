import numpy as np
import cv2
import math


# Return a list of lines [ [x1,y1,x2,y2], ...]
def connectSegments(segmentList, threshold = 0):
    lines = []
    segmentList.sort(key=lambda x: x[0])
    xmin = 0
    i = 0
    N = len(segmentList)
    while i < N:
        newline = segmentList[i]
        if newline[0] > xmin:
            j = i+1
            while j < N:
                x2_sg1 = newline[2]
                x1_sg2 = segmentList[j][0]
                x2_sg2 = segmentList[j][2]
                y2_sg2 = segmentList[j][3]
                if x2_sg1 >= x1_sg2 - threshold:
                    if x2_sg1 < x2_sg2:
                        newline[2] = x2_sg2
                        newline[3] = y2_sg2
                else:
                    break
                j = j+1
            xmin = newline[2]
            lines.append(newline)
        else:
            i = i+1
    return lines
# Return a list of line segments by theta and radius
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
                radius_SG = lineSegmentGroups[i][0]
                theta_SG = lineSegmentGroups[i][1]

                if abs(radius - radius_SG) < thresholdRadius and abs(theta - theta_SG) < thresholdTheta:
                    lineSegmentGroups[i][2].append([x1,y1,x2,y2])
                    addNewLine = False
                    break
            if addNewLine:
                lineSegmentGroups.append([radius, theta, [[x1, y1, x2, y2]]])
        else:
            lineSegmentGroups.append([radius, theta, [[x1, y1, x2, y2]]])
    return lineSegmentGroups
# Return a list of filtered lines. Per line: [x1,y1,x2,y2,radius,theta]
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
# Return list of [x, y] coordines of all intersecing lines
def findAllIntersections(lines):
    intersections =[]
    N = len(lines)
    for i in range(N):
        line1 = extendLine(lines[i][0:4])
        for j in range(i+1,N):
            line2 = extendLine(lines[j][0:4])
            if intersect(line1, line2):
                intersections.append(findIntersection(line1, line2))
    return intersections
# Return [x, y] coordinates of intersection between line1 and line2
def findIntersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return Px, Py
# Called from intersect
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
# Return true if line segments A and B intersect
def intersect(line1, line2):
    A = [line1[0],line1[1]]
    B = [line1[2],line1[3]]
    C = [line2[0],line2[1]]
    D = [line2[2],line2[3]]
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# Return line [x1,y2,x2,y2] with extended end points
def extendLine(line, extendPixels = 500.0):
    x1, y1, x2, y2 = line
    length = math.sqrt((x1-x2)**2.0 + (y1-y2)**2.0)
    dx = int((extendPixels/length) * abs(x1-x2))
    dy = int((extendPixels/length) * abs(y1-y2))
    x1 = x1 - dx
    x2 = x2 + dx
    if y1 < y2:
        y1 = y1 - dy
        y2 = y2 + dy
    else:
        y1 = y1 + dy
        y2 = y2 - dy
    return [x1,y1,x2,y2]
# Return list [[[x1,y1],[x2,y2],[x3,y3],[x4,y5]], ...] of four cornerpoints
def findContoursOld(lines):
    cornerPoints =[]
    N = len(lines)
    for i in range(N):
        line1 = lines[i][:4][0]
        for j in (lineN for lineN in range(i+1,N) if lineN != i):
            line2 = lines[j][:4][0]
            if intersect(line1,line2):
                p1 = findIntersection(line1,line2)
                for k in (lineN for lineN in range(i+1,N) if lineN not in (i,j)):
                    line3 = lines[k][:4][0]
                    if intersect(line2, line3) and not intersect(line1, line3):
                        p2 = findIntersection(line2, line3)
                        for l in (lineN for lineN in range(i+1,N) if lineN not in (i,j,k)):
                            line4 = lines[l][:4][0]
                            if (intersect(line3, line4) and intersect(line1, line4)) and not intersect(line2, line4):
                                p3 = findIntersection(line3,line4)
                                p4 = findIntersection(line4,line1)
                                cornerPoints.append([p1,p2,p3,p4])
    return cornerPoints
#def findContourCornerPoints():

# Set target scene
target_scene = cv2.imread('Figures/Screenshot.jpg', 0)
#target_scene_original = target_scene.copy()
#target_scene_update = target_scene.copy()

# Filter image (Canny>Dilate>Erode>Canny)
kernel = np.ones((10, 10), np.uint8)
edges = cv2.Canny(target_scene, 20, 100)
#cv2.imshow('Canny', edges)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Dilate+Erode', edges)
edges = cv2.Canny(edges, 20, 100)
cv2.imshow('Canny 2', edges)
#cv2.waitKey(0)

# Get contours, filter small contour areas
im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contoursFiltered = sorted([x for x in contours if cv2.contourArea(x) > 4000], key=lambda x: cv2.contourArea(x), reverse=True)

# Per contour: Create Houghlines and compute intersection points
i=1
for cntr in contoursFiltered[:5]:
    blank_image = np.zeros(target_scene.shape, np.uint8)
    cv2.drawContours(blank_image, cntr, -1, 255, 10)
    lines = cv2.HoughLinesP(blank_image, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=30, maxLineGap=30)
    filteredLines = findLines(lines)
    intersections = findAllIntersections(filteredLines)

    # Show image with contour and cornerpoints
    target_scene_update = target_scene.copy()
    for line in filteredLines:
        x1,y1,x2,y2 = line[:4]
        target_scene_update = cv2.line(target_scene_update, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for xy in intersections:
        cv2.circle(target_scene_update, (int(xy[0]), int(xy[1])), radius=10, color=0, thickness=-1)
    cv2.imshow(str(i),target_scene_update)
    i=i+1

cv2.waitKey(0)
# Remove small areas and sort by area
#cntsSorted = sorted([x for x in contours if cv2.contourArea(x) > 4000], key=lambda x: cv2.contourArea(x) )
#cntsSorted = [x for x in cntsSorted if cv2.contourArea(x) > 4000]



#print('len',len(contours),len(filteredLines))

#blank_image = np.zeros(target_scene.shape, np.uint8)
#for line in filteredLines:
#    x1,y1,x2,y2 = line[:4]
#    blank_image = cv2.line(blank_image, (x1, y1), (x2, y2), 255, 10)
#im2, contours, hierarchy = cv2.findContours(blank_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contoursNew = [x for x in contours if len(x) <= 2000]
#cv2.drawContours(target_scene_update, contoursNew, -1, 255, 3)

#for line in filteredLines:
#    x1,y1,x2,y2 = line[:4]
#    target_scene_update = cv2.line(target_scene_update, (x1, y1), (x2, y2), (255, 0, 0), 2)

#for line in lines:
#    x1, y1, x2, y2 = line[0][:4]
#    target_scene = cv2.line(target_scene, (x1, y1), (x2, y2), (255, 0, 0), 2)

if True:
    cv2.imshow('Canny edges',target_scene)
    cv2.imshow('contours',blank_image)
    cv2.imshow('result',target_scene_update)
    cv2.waitKey(0)