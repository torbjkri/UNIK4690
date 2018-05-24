import cv2
import numpy as np
import math

# return [float, float]
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

# Return filtered gray image (np.array)
def filterImage(image, actions = [1,2], kernel1 = np.ones((3,3), np.uint8), kernel2 = (3,3)):
    for action in actions:
        if action == 1:
            image = cv2.GaussianBlur(image, kernel2, 0.7)
        elif action == 2:
            image = cv2.Canny(image, threshold1=30, threshold2=60)
        elif action == 3:
            image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel1, iterations=2)
        elif action == 4:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1, iterations=1)
        elif action == 5:
            image = cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((2,2), np.uint8), iterations=1)
        elif action == 6:
            image = cv2.dilate(image, kernel1)
    return image

# Return coordinates as integer [x_cp, y_cp]
def VanishingPoint(edgeImage,orientation = 'diagonal'):
    th_A = 0.05*np.pi
    th_B = 0.45*np.pi
    th_C = 0.55 * np.pi
    th_D = 0.95 * np.pi
    th_E = 0.48 * np.pi
    th_F = 0.52 * np.pi
    th_G = 0.02 * np.pi
    th_H = 0.98 * np.pi

    # Define upper and lower angle limits for houghline detection
    iterations = 0
    nrOfHlines = None
    t = 250
    while nrOfHlines is None or 4 > nrOfHlines or nrOfHlines > 30:
        if iterations == 30:
            print('max nr of iterations reached for '+orientation)
            return None
        elif nrOfHlines is None or nrOfHlines < 4:
            t -= 10
            if t < 100:
                return None
        else:
            t += 25
        iterations += 1
        print('hlines detected: ', nrOfHlines, 't=', t, 'iteration:', iterations, orientation)
        # Step 1. Get Hough lines
        if orientation == 'horizontal':
            hlines = cv2.HoughLines(edgeImage, rho=1, theta=1 * np.pi / 180, threshold=t, min_theta=th_E,
                                max_theta=th_F)
            if hlines is not None:
                nrOfHlines = len(hlines)
        elif orientation == 'vertical':
            hlines = cv2.HoughLines(edgeImage, rho=1, theta=1 * np.pi / 180, threshold=t)
            min_theta = 0.45 * np.pi
            max_theta = 0.55 * np.pi
            if hlines is not None:
                hlines = [x for x in hlines if (x[0][1] <  th_G or x[0][1] > th_H)]
                if hlines is not None:
                    nrOfHlines = len(hlines)
        elif orientation == 'diagonal':

            hlines = cv2.HoughLines(edgeImage,rho=1, theta=1 * np.pi / 180,threshold=t,min_theta=th_A,max_theta=th_D)
            if hlines is not None:
                hlines_down = [x for x in hlines if x[0][1] < th_B]
                hlines_up = [x for x in hlines if x[0][1] > th_C]
                if len(hlines_down) !=0 and len(hlines_up) != 0:
                    nrOfHlines = len(hlines_down)+len(hlines_up)
        else:
            return None
    # Step 2. compute intersections inx of lines
    inxs = []
    if orientation in ['vertical', 'horizontal']:
        hlinesCartesian = []
        for line in hlines:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x1 = rho*a
            y1 = rho*b
            line = [x1, y1, x1 - b, y1 + a]
            for line2 in hlinesCartesian:
                inx = findIntersection(line, line2)
                if inx is not None and abs(inx[0]) < 1000000 and abs(inx[1]) < 1000000:
                    inxs.append(findIntersection(line, line2))
            hlinesCartesian.append(line)
    else:
        hlines_upCartesian = []
        for line in hlines_up:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x1 = rho * a
            y1 = rho * b
            hlines_upCartesian.append([x1, y1, x1 - b, y1 + a])
        for line in hlines_down:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x1 = rho * a
            y1 = rho * b
            for line2 in hlines_upCartesian:
                inxs.append(findIntersection([x1, y1, x1 - b, y1 + a], line2))

    # Compute vanishing point vp from intersections
    if len(inxs) > 0:
        inxs = [x for x in inxs if x is not None]
        if orientation == 'vertical':
            #y = [x[1] for x in inxs]
            y_abs = [abs(x[1]) for x in inxs]
            vp = [x for _,x in sorted(zip(y_abs,inxs), key=lambda pair: pair[0])][-1]
        elif orientation == 'horizontal':
            x_abs = [abs(x[0]) for x in inxs]
            vp = [x for _,x in sorted(zip(x_abs,inxs), key=lambda pair: pair[0])][-1]
        else:
            dist_sums = []
            for ii, inx in enumerate(inxs):
                dist_sum = 0
                for jj in [x for x in range(len(inxs)) if x != ii]:
                    dist_sum = dist_sum + math.sqrt((inx[0]-inxs[jj][0])**2 + (inx[1]-inxs[jj][1])**2)
                    dist_sums.append(dist_sum)
            vp = [x for _, x in sorted(zip(dist_sums,inxs), key=lambda pair: pair[0])][0]
            # cv2.drawMarker(edgeImage,(int(vanishingPoint[0]),int(vanishingPoint[1])),123,markerSize=255)
            #cv2.imshow('name',edgeImage)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        return [int(vp[0]), int(vp[1])]
    else:
        return None

def VanishingPointP(edgeImage):
       # Define upper and lower angle limits for houghline detection
    iterations = 0
    nrOfHlines = None
    t = 250
    while nrOfHlines is None or 30 > nrOfHlines or nrOfHlines > 50:
        if iterations == 30:
            return None
        elif nrOfHlines is None or nrOfHlines < 30:
            t -= 20
            if t < 50:
                return None
        else:
            t += 25
        iterations += 1
        print('hlines detected: ', nrOfHlines, 't=', t, 'iteration:', iterations)
    # Step 1. Get Hough lines
        hlines = cv2.HoughLinesP(edgeImage, rho=1, theta=1 * np.pi / 180, threshold=t, minLineLength=20, maxLineGap=200)
        if hlines is not None:
            hlines_horizontal = []
            hlines_vertical = []
            hlines_diagonal_up = []
            hlines_diagonal_down = []
            nrOfHlines = len(hlines)
            for line in hlines:
                dx = (line[0][2] - line[0][0])
                dy = (line[0][3] - line[0][1])
                if dx == 0:
                    hlines_vertical.append(line[0])
                else:
                    slope = dy/dx
                    abs_slope = abs(slope)
                    if abs_slope < 0.1:
                        hlines_horizontal.append(line[0])
                    elif abs_slope > 10:
                        hlines_vertical.append(line[0])
                    elif 0.2 < slope < 5:
                        hlines_diagonal_up.append(line[0])
                    elif -5 < slope < -0.2:
                        hlines_diagonal_down.append(line[0])
        else:
            return None

    # Step 2. compute intersections inx of lines
    inxs_horizontal = []
    inxs_vertical = []
    inxs_diagonal = []
    for ii in range(len(hlines_horizontal)):
        for jj in [x for x in range(len(hlines_horizontal)) if x > ii]:
            inx = findIntersection(hlines_horizontal[ii], hlines_horizontal[jj])
            if inx is not None:
                inxs_horizontal.append(inx)
    for ii in range(len(hlines_vertical)):
       for jj in [x for x in range(len(hlines_vertical)) if x > ii]:
           inx = findIntersection(hlines_vertical[ii], hlines_vertical[jj])
           if inx is not None:
               inxs_vertical.append(inx)
    for line1 in hlines_diagonal_up:
        for line2 in hlines_diagonal_down:
            inx = findIntersection(line1,line2)
            if inx is not None:
                inxs_diagonal.append(inx)

    # Compute vanishing point vp from intersections
    vp_v = None
    vp_h = None
    vp_d = None
    for orientation in ['v','h','d']:
        if orientation == 'v' and len(inxs_vertical) != 0:
            y_abs = [abs(x[1]) for x in inxs_vertical]
            vp_v = [x for _, x in sorted(zip(y_abs, inxs_vertical), key=lambda pair: pair[0])][-1]
        elif orientation == 'h' and len(inxs_horizontal) != 0:
            x_abs = [abs(x[0]) for x in inxs_horizontal]
            vp_h = [x for _,x in sorted(zip(x_abs,inxs_horizontal), key=lambda pair: pair[0])][-1]
        elif orientation == 'd':
            if len(inxs_diagonal) != 0:
                dist_sums = []
                for ii, inx in enumerate(inxs_diagonal):
                    dist_sum = 0
                    for jj in [x for x in range(len(inxs_diagonal)) if x != ii]:
                        dist_sum = dist_sum + math.sqrt((inx[0]-inxs_diagonal[jj][0])**2 + (inx[1]-inxs_diagonal[jj][1])**2)
                        dist_sums.append(dist_sum)
                vp_d = [x for _, x in sorted(zip(dist_sums,inxs_diagonal), key=lambda pair: pair[0])][0]
            else:
                vp_d = vp_h
            # cv2.drawMarker(edgeImage,(int(vanishingPoint[0]),int(vanishingPoint[1])),123,markerSize=255)
            #cv2.imshow('name',edgeImage)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    return [vp_v, vp_h, vp_d, t]

# Return cornerpoints from vansishing points as np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
def computeDstCornerpoints(image, cp, side = 'left', cpV = None, cpH = None,
                           d_edge = 300,d_cp = 300,
                           d_left = 0, d_right = 0,
                           d_top = 0, d_bottom = 0):
    height, width = image.shape
    if cp is None:
        if side in ['left', 'right'] and cpH is not None:
            cp_x = cpH[0]
            cp_y = cpH[1]
        elif side in ['top','bottom'] and cpV is not None:
            cp_x = cpV[0]
            cp_y = cpV[1]
        else:
            return None
    else:
        cp_x = cp[0]
        cp_y = cp[1]

    # read side (top, bottom, left or right and corresponding calculations
    if side == 'left':
        slope1 = (cp_y-d_top)/cp_x
        x1 = d_edge
        x2 = cp_x-d_cp
        pnt1 = [x1, d_top+slope1*x1]
        pnt2 = [x2, d_top+slope1*x2]
        if cpV is None:
            slope2 = (cp_y - height + d_top) / cp_x
            pnt3 = [x2, (height - d_bottom) + slope2 * x2]
            pnt4 = [x1, (height - d_bottom) + slope2 * x1]
        else:
            pnt3 = findIntersection([x2,pnt2[1],cpV[0],cpV[1]],[0,height-d_bottom, cp_x,cp_y])
            pnt4 = findIntersection([x1,pnt1[1],cpV[0],cpV[1]],[0,height-d_bottom, cp_x,cp_y])
    elif side == 'right':
        slope1 = (d_top-cp_y)/(width-cp_x)
        x1 = cp_x+d_cp
        x2 = width-d_edge
        pnt1 = [x1, cp_y+slope1*(x1-cp_x)]
        pnt2 = [x2, cp_y+slope1*(x2-cp_x)]
        if cpV is None:
            slope2 = (height - cp_y - d_top) / (width - cp_x)
            pnt3 = [x2, cp_y + slope2 * (x2-cp_x)]
            pnt4 = [x1, cp_y + slope2 * (x1-cp_x)]
        else:
            pnt3 = findIntersection([x2,pnt2[1],cpV[0],cpV[1]],[width,height-d_bottom, cp_x,cp_y])
            pnt4 = findIntersection([x1,pnt1[1],cpV[0],cpV[1]],[width,height-d_bottom, cp_x,cp_y])
    elif side == 'bottom':
        slope1 = (cp_y-height)/(cp_x-d_left)
        y4 = height - d_edge
        y1 = cp_y + d_cp
        pnt1 = [cp_x - (cp_y- y1)/slope1,y1]
        pnt4 = [cp_x - (cp_y- y4)/slope1,y4]
        if cpH is None:
            slope2 = (height-cp_y)/(width+d_right-cp_x)
            pnt2 = [cp_x + slope2 * (y1 - cp_y),y1]
            pnt3 = [cp_x + slope2 * (y4 - cp_y),y4]
        else:
            pnt2 = findIntersection([pnt1[0],y1,cpH[0],cpH[1]],[width-d_right,height, cp_x,cp_y])
            pnt3 = findIntersection([pnt4[0],y4,cpH[0],cpH[1]],[width-d_right,height, cp_x,cp_y])
    elif side == 'top':
        slope1 = cp_y/(cp_x-d_left)
        y4 = cp_y - d_cp
        y1 = d_edge
        pnt1 = [cp_x - (cp_y- y1)/slope1,y1]
        pnt4 = [cp_x - (cp_y- y4)/slope1,y4]
        if cpH is None:
            slope2 = (-cp_y)/(width+d_right-cp_x)
            pnt2 = [cp_x + slope2 * (y1 - cp_y),y1]
            pnt3 = [cp_x + slope2 * (y4 - cp_y),y4]
        else:
            pnt2 = findIntersection([pnt1[0],y1,cpH[0],cpH[1]],[width-d_right,0, cp_x,cp_y])
            pnt3 = findIntersection([pnt4[0],y4,cpH[0],cpH[1]],[width-d_right,0, cp_x,cp_y])
    else:
        return None
    pts_dst = np.array([pnt1, pnt2, pnt3, pnt4])
    return pts_dst

# Return gray image (np.array)
def projectImage(img1, img2, homography):

    w2, h2 = img1.shape
    # create mask and inverse mask
    mask = cv2.warpPerspective(np.ones(img2.shape), homography, dsize=(h2, w2),)
    maskInverted = np.ones(img1.shape) - mask
    # warp image to be projected
    img2_warped = cv2.warpPerspective(img2, homography, dsize=(h2, w2))

    # multiply img2 with inverted mask
    img1_bg = np.asanyarray(np.multiply(img1, maskInverted), dtype='uint8')

    # add images
    result = img1_bg+img2_warped

    return result

# return edge coordinates of diagonal with largest theta from origin
def findMainDiagonal(image):
    h, w = image.shape
    cv2.imshow('test', image)
    cv2.waitKey(0)
    lines = cv2.HoughLines(image,rho=3, theta=3 * np.pi / 180, threshold=200,max_theta=0.5*np.pi)
    if lines is not None:
        thetas = [x[0][1] for x in lines if abs(x[0][0]) < 50]
        if len(thetas) != 0:
            theta = max(thetas)
            y = w * math.tan(theta)
            image = cv2.line(image,(0,0),(w,int(y)),122,thickness=20)
            cv2.imshow('ressss', image)
            cv2.waitKey(0)
            if y > h:
                x = h * math.tanh(theta)
                return [h / math.tan(theta),h]
            else:
                return [w,y]
    return None

def findMainDiagonalP(image):
    h, w = image.shape
    cornerthreshold = 30
    t = 200
    t_slope = 0
    edgepoint = [w,h]
    while edgepoint[0] == w and edgepoint[1] == h:
        if t < 50:
            break
        hlines = cv2.HoughLinesP(image, rho=1, theta=1 * np.pi / 180, threshold=t, minLineLength=50, maxLineGap=50)
        if hlines is not None:
            for line in hlines:
                x1,y1,x2,y2 = line[0]
                if x1 != x2 and y1 < y2:
                    slope = (y2-y1)/(x2-x1)
                    if 0.2 < slope < 5:
                        y_at_x0 = y1 - slope * x1
                        x_at_y0 = - y_at_x0/ slope
                        if (y_at_x0 < cornerthreshold or x_at_y0 < cornerthreshold) and slope > t_slope:
                                t_slope = slope
                                y_at_w = y_at_x0 + slope * w
                                x_at_h = (h-y_at_x0)/slope
                                if y_at_w > h:
                                    edgepoint = [x_at_h, h]
                                else:
                                    edgepoint = [w, y_at_w]
        t -= 20
    return edgepoint

def findAllDiagonal(image,vp,corner):
    vpX = int(vp[0])
    vpY = int(vp[1])
    if corner == 'bl':
        M = cv2.flip(image[vpY:,:vpX],1)
        dxy = findMainDiagonalP(M)
        x = vpX - dxy[0]
        y = vpY + dxy[1]
    elif corner == 'tr':
        M=cv2.flip(image[:vpY,vpX:],0)
        dxy = findMainDiagonalP(M)
        x = vpX + dxy[0]
        y = vpY - dxy[1]
    elif corner == 'tl':
        M = cv2.flip(cv2.flip(image[:vpY,:vpX],0),1)
        dxy = findMainDiagonalP(M)
        x = vpX - dxy[0]
        y = vpY - dxy[1]
    elif corner == 'br':
        M = image[vpY:,vpX:]
        dxy = findMainDiagonalP(M)
        x = vpX + dxy[0]
        y = vpY + dxy[1]
    return [int(x),int(y)]


def mainFunction(imageName = 'Figures/corridor.jpg', projectionName = 'Figures/banksy.jpg', side = 'left'):
    # 1. Read image and convert to grayscale
    original = cv2.imread(imageName)
    projection = cv2.imread(projectionName)
    originalGrey = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
    projectionGrey = cv2.cvtColor(projection.copy(), cv2.COLOR_BGR2GRAY)
    originalWithProjection = originalGrey.copy()

    # 2. Get corner points of target projection for homography calculation (step 5)
    heightP, widthP = projectionGrey.shape
    pts_src = np.array([[0, 0], [widthP, 0], [widthP, heightP], [0, heightP]])

    # 3. compute filtered image with 1pxl lines (canny)
    edges = filterImage(originalGrey)
    #cv2.imshow('Filter result', edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 4. Find vanishing point of the non-vertical lines
    #vpD = VanishingPoint(edges, orientation='diagonal')
    #print(vpD)
    #vpV = None
    #vpH = None
    #if vpD is None:
    #    return None
    #vpV = VanishingPoint(edges, orientation='vertical')
    #vpH = VanishingPoint(edges, orientation='horizontal')

    vpV,vpH,vpD,t = VanishingPointP(edges)
    if vpH is None and vpD is None:
        vpD = [500,500]
    # 5.A find cornerpoints
    pts_dst_method = 1
    R = originalGrey
    G = originalGrey
    B = originalGrey
    h0, w0 = originalGrey.shape
    pts_src_grey = np.array([[0, 0], [w0, 0], [w0,h0], [0, h0]])
    for side in ['left','right','top','bottom']:
        if pts_dst_method == 1:
            if side == 'left':
            # LEFT WALL:
                pnt1 = findAllDiagonal(edges,vpD,'tl')
                pnt4 = findAllDiagonal(edges,vpD,'bl')
                pnt3 = [vpD[0],vpD[1]+5]
                pnt2 = [vpD[0],vpD[1]-5]
            elif side == 'right':
            # RIGHT WALL
                pnt2 = findAllDiagonal(edges,vpD,'tr')
                pnt3 = findAllDiagonal(edges,vpD,'br')
                pnt4 = [vpD[0],vpD[1]+5]
                pnt1 = [vpD[0],vpD[1]-5]
            elif side == 'top':
            # ceiling
                pnt1 = findAllDiagonal(edges,vpD,'tl')
                pnt2 = findAllDiagonal(edges,vpD,'tr')
                pnt3 = [vpD[0]+5,vpD[1]]
                pnt4 = [vpD[0]-5,vpD[1]]
            elif side == 'bottom':
            # floor
                pnt4 = findAllDiagonal(edges,vpD,'bl')
                pnt3 = findAllDiagonal(edges,vpD,'br')
                pnt1 = [vpD[0]-5,vpD[1]]
                pnt2 = [vpD[0]+5,vpD[1]]
            pts_dst = np.array([pnt1,pnt2,pnt3,pnt4])
        else:
        # 5.B find corner points
            pts_dst = computeDstCornerpoints(originalGrey,
                                                 vpD,
                                                 cpV=vpV,
                                                 cpH=vpH,
                                                 side = side,
                                                 d_edge = 50,
                                                 d_cp = 50,
                                                 d_left = 0,
                                                 d_right = 0,
                                                 d_top = 0,
                                                 d_bottom = 0)
        if pts_dst is not None:

        # 6. find hompography
            homography, status = cv2.findHomography(pts_src, pts_dst)


        # convert greyscale to color and project color on target side
            homography, status = cv2.findHomography(pts_src_grey, pts_dst)
            #colorimage = cv2.cvtColor(originalGrey,cv2.COLOR_GRAY2BGR)
            mask = np.ones(originalGrey.shape) - cv2.warpPerspective(np.ones(originalGrey.shape), homography,                                                     dsize=(w0, h0))
            if side == 'top':
                R = np.asanyarray(np.multiply(R, mask), dtype='uint8')
            elif side == 'bottom':
                G = np.asanyarray(np.multiply(G, mask), dtype='uint8')
            else:
                B = np.asanyarray(np.multiply(B, mask), dtype='uint8')
    res = cv2.merge([R,G,B])
    cv2.imshow(side,res)
    cv2.waitKey(0)


    #originalWithProjection = projectImage(originalWithProjection, projectionGrey, homography)
    #cv2.imshow('projection', originalWithProjection)
    #cv2.waitKey(0)

    # Generate new QC code
    #img_QC = cv2.imread('Figures/QR_real.png',0)
    #h0, w0 = originalGrey.shape
    #h1, w1 = img_QC.shape
    #pts_src = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    #homography, status = cv2.findHomography(pts_src, pts_dst)
    #img_QC = cv2.warpPerspective(img_QC, homography, dsize=(w0,h0))
    #cv2.imshow('Targeting wall on the '+side+str(vpD)+str(vpV)+str(vpH), originalWithProjection)
    #cv2.imshow('QR code '+str(vpD)+str(vpV)+str(vpH), img_QC)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return originalWithProjection



imageList = [
'hallway2',
    'corridor',
    'corridor1',
    'corridor2',
    'corridor4',
    'hallway4',
    #'hallway5',
]

for name in imageList:
    image_result = mainFunction('Figures/'+name+'.jpg', side = 'removeMe')
cv2.destroyAllWindows()
