import cv2
import numpy as np
import networkx as nx
import segmentation_functions as sf
import projection as pr
import math
import ProjectionFromContours as pfc

def findIntersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    division = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    if division != 0:
        Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
             division
        Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
             division
        return [Px, Py]
    else:
        return None

# Return filtered image
def filterImage(image, actions = [1,2], kernel1 = np.ones((2,2), np.uint8), kernel2 = (3,3)):
    for action in actions:
        if action == 1:
            image = cv2.GaussianBlur(image, kernel2, 0.5)
        elif action == 2:
            image = cv2.Canny(image, threshold1=20, threshold2=50)
        elif action == 3:
            image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel1, iterations=2)
        elif action == 4:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1, iterations=1)
        elif action == 5:
            image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1, iterations=2)
        elif action == 6:
            image = cv2.dilate(image, kernel1)
    cv2.imshow('Filter result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

# Return [[int,int],int,int], coordinates, number of lines up, number of lines down. Otherwise 'None'
def VanishingPoint(edgeImage,th_min = 0.05*np.pi, th_max = 0.95*np.pi, th_min2 = None, th_max2 = None):
    # Get Houghlines except vertical lines
    nrOfHlines = 21
    t=80
    while nrOfHlines > 20:
        t=t+20
        hlines = cv2.HoughLines(edgeImage,rho=1, theta=1 * np.pi / 180,threshold=t,min_theta=th_min,max_theta=th_max)
        if hlines is not None and (th_min2 is not None and th_max2 is not None):
            hlines = [x for x in hlines if (x[0][1] < th_min2 or th_max2 < x[0][1])]
        if hlines is None:
            break
        nrOfHlines = len(hlines)

    if hlines is not None:
        # Convert to cartesian coordinates and group by up-and down going lines
        hlinesCartesianUp = []
        hlinesCartesianDown = []
        for line in hlines:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pnt1 = np.array([int(x0 + 1000 * -b), int(y0 + 1000 * a)])
            pnt2 = np.array([int(x0 - 10000 * -b), int(y0 - 10000 * a)])
            if theta < np.pi/2:
                hlinesCartesianUp.append([pnt1[0],pnt1[1],pnt2[0],pnt2[1]])
            else:
                hlinesCartesianDown.append([pnt1[0],pnt1[1],pnt2[0],pnt2[1]])
            cv2.line(edgeImage,(pnt1[0],pnt1[1]),(pnt2[0],pnt2[1]),255,thickness=1)
        # compute vanishing point if there are both up and down going lines.
        if len(hlinesCartesianDown) > 0 and len(hlinesCartesianUp) > 0:
            intersections = []
            for ii, line1 in enumerate(hlinesCartesianDown):
                for line2 in hlinesCartesianUp:
                    intersection = findIntersection(line1,line2)
                    intersections.append(intersection)
                    cv2.drawMarker(edgeImage, (int(intersection[0]), int(intersection[1])), 255, markerSize=10)
            sumOfDistances = []
            for ii, intersection in enumerate(intersections):
                sumOfDistance = 0
                for jj in [x for x in range(len(intersections)) if x != ii]:
                    sumOfDistance = sumOfDistance + math.sqrt((intersection[0]-intersections[jj][0])**2 + (intersection[1]-intersections[jj][1])**2)
                sumOfDistances.append(sumOfDistance)
            vanishingPoint = [x for _, x in sorted(zip(sumOfDistances,intersections), key=lambda pair: pair[0])][0]
            cv2.drawMarker(edgeImage,(int(vanishingPoint[0]),int(vanishingPoint[1])),123,markerSize=255)
            #cv2.imshow('name',edgeImage)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            return [int(vanishingPoint[0]), int(vanishingPoint[1]), len(hlinesCartesianDown), len(hlinesCartesianUp)]
    return None

def computeDstCornerpoints(image,VP_diagonal,VP_vertical = None,dx_top = 0,dx_bottom = 0,x_indentFromEdge = 0,x_indentFromVP = 20):
    heightO, widthO = image.shape
    if x_indentFromVP == 0:
        x_indentFromVP = 20
    slope = VP_diagonal[1] / (VP_diagonal[0] - dx_top)
    pnt1 = [x_indentFromEdge, int(slope * (dx_top + x_indentFromEdge - VP_diagonal[0]) + VP_diagonal[1])]
    pnt2 = [VP_diagonal[0] - x_indentFromVP, int(-slope * x_indentFromVP + VP_diagonal[1])]
    VP_vertical = VP_vertical
    if VP_vertical is None:
        pnt3 = findIntersection([pnt2[0], pnt2[1], pnt2[0], pnt2[1] + 100],
                                [dx_bottom, heightO, VP_diagonal[0], VP_diagonal[1]])
        pnt4 = findIntersection([pnt1[0], pnt1[1], pnt1[0], pnt1[1] + 100],
                                [dx_bottom, heightO, VP_diagonal[0], VP_diagonal[1]])
    else:
        pnt3 = findIntersection([pnt2[0], pnt2[1], VP_vertical[0], VP_vertical[1]],
                                [dx_bottom, heightO, VP_diagonal[0], VP_diagonal[1]])
        pnt4 = findIntersection([pnt1[0], pnt1[1], VP_vertical[0], VP_vertical[1]],
                                [dx_bottom, heightO, VP_diagonal[0], VP_diagonal[1]])

    pts_dst = np.array([pnt1, pnt2, pnt3, pnt4])
    return pts_dst

def projectImage(imageName = 'Figures/corridor.jpg', projectionName = 'Figures/banksy.jpg'):
    # 1. Read image and convert to grayscale
    original = cv2.imread(imageName)
    projection = cv2.imread(projectionName)
    originalGrey = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
    projectionGrey = cv2.cvtColor(projection.copy(), cv2.COLOR_BGR2GRAY)
    originalWithProjection = originalGrey.copy()

    # 2. Get corner points of target projection for homography calculation (step 5)
    heightP, widthP = projectionGrey.shape
    pts_src = np.array([[0, 0], [widthP, 0], [widthP, heightP], [0, heightP]])
    imageCenter=[int(widthP/2),int(heightP/2)]

    # 3. compute filtered image with 1pxl lines (canny)
    edges = filterImage(originalGrey)

    # 4. Find vanishing point of the non-vertical lines
    VP_diagonal = VanishingPoint(edges, th_min2=0.45*np.pi, th_max2=0.55*np.pi)
    VP_vertical = VanishingPoint(edges, th_min=0, th_max=np.pi, th_min2=0.1*np.pi, th_max2=0.9*np.pi)
    # 5. Compute homography assuming the wall takes 60% (counting from the middle) of the image edge
    if VP_diagonal is not None:
        if VP_diagonal[0] <= imageCenter[0]:
            print('Targeting wall on the right')
            side = 'right (flipped image)'
            edges = cv2.flip(edges,0)
            originalGrey = cv2.flip(originalGrey,0)
            originalWithProjection = cv2.flip(originalWithProjection,0)
            projectionGrey = cv2.flip(projectionGrey,0)
            VP_diagonal = VanishingPoint(edges, th_min2=0.45 * np.pi, th_max2=0.55 * np.pi)
            VP_vertical = VanishingPoint(edges, th_min=0, th_max=np.pi, th_min2=0.1 * np.pi, th_max2=0.9 * np.pi)
            pts_dst = computeDstCornerpoints(originalGrey,VP_diagonal,VP_vertical)

        else:
            print('Targeting wall on the left')
            side = 'left'
            pts_dst = computeDstCornerpoints(originalGrey,VP_diagonal,VP_vertical)

        homography, status = cv2.findHomography(pts_src, pts_dst)

        originalWithProjection = pr.projectImage(originalWithProjection, projectionGrey, homography)
        cv2.imshow('Targeting wall on the '+side, originalWithProjection)
        cv2.waitKey(0)

        # Generate new QC code
        img_QC = cv2.imread('Figures/QR_real.png',0)
        width = max([x[0] for x in pts_dst])
        height = max([x[1] for x in pts_dst])
        blank_image = np.zeros((int(height),int(width)), np.uint8)
        blank_image = pr.projectImage(blank_image, img_QC, homography)
        cv2.imshow('QR code', blank_image)
        cv2.waitKey(0)
    return originalWithProjection

imageList = ['hallway2',
             'hallway3',
             'hallway4']

for name in imageList:
    image_result = projectImage('Figures/'+name+'.jpg')
cv2.destroyAllWindows()