import numpy as np
import cv2
import camera as cm
import projection as pr
import math
from matplotlib import pyplot as plt

def connectSegments(segmentList):
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
                if x2_sg1 >= x1_sg2 - 10:
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

def groupSegments(lines):
    lineSegmentGroups = []
    for line in lines:
        x1,y1,x2,y2 = line[0]

        # compute theta (-0.5pi to 0.5 pi) and radius (negative and positive)'
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
        rx = radius * -math.sin(theta)
        ry = radius * math.cos(theta)


        # check if theta and radius fall in previous group, otherwise create new group
        N = len(lineSegmentGroups)
        addNewLine = True
        if N > 0:
            for i in range(N):
                rx_SG = lineSegmentGroups[i][0]
                ry_SG = lineSegmentGroups[i][1]
                if math.sqrt((rx - rx_SG)**2 + (ry - ry_SG)**2) < 20:
                    lineSegmentGroups[i][2].append([x1,y1,x2,y2])
                    addNewLine = False
                    break
            if addNewLine:
                lineSegmentGroups.append([rx, ry, [[x1, y1, x2, y2]], radius, theta,gradient])
        else:
            lineSegmentGroups.append([rx, ry, [[x1, y1, x2, y2]], radius, theta,gradient])
    return lineSegmentGroups

def findLines(lines):
    final_lines=[]
    final_data=[]
    lineSegmentGroups = groupSegments(lines)
    for lineSegmentGroup in lineSegmentGroups:
        connectedlines = (connectSegments(lineSegmentGroup[2]))
        for connectedline in connectedlines:
            final_lines.append(connectedline)

    return final_lines


target_scene = cv2.imread('Figures/Screenshot.jpg', 0)
img = target_scene.copy()
edges = cv2.Canny(target_scene, 20, 200)
lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=150, maxLineGap=100)

filteredlines = findLines(lines)

for line in filteredlines:
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

for line in lines:
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]
    target_scene = cv2.line(target_scene, (x1, y1), (x2, y2), (255, 0, 0), 2)

print(len(filteredlines),len(lines))
if True:
    #cv2.imshow('original',target_scene)
    cv2.imshow('result',img)
    cv2.waitKey(0)