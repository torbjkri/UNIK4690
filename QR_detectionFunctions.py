import cv2
import numpy as np
import math

x_coord, y_coord = -1, -1
mouseClick = False


def projectPoint(event, x, y, flags, param):
    global x_coord, y_coord, mouseClick
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseClick = True
        x_coord = x
        y_coord = y

# Return gray image (np.array)
def projectImage(img1_grey, img2_grey, homography):
    w2, h2 = img1_grey.shape
    # create mask and inverse mask
    mask = cv2.warpPerspective(np.ones(img2_grey.shape), homography, dsize=(h2, w2),)
    maskInverted = np.ones(img1_grey.shape) - mask
    # warp image to be projected
    img2_warped = cv2.warpPerspective(img2_grey, homography, dsize=(h2, w2))
    # multiply img2 with inverted mask
    img1_bg = np.asanyarray(np.multiply(img1_grey, maskInverted), dtype='uint8')
    # add images
    edges = img1_bg + img2_warped
    cv2.imwrite('Figures/LaTex_images/filter_corridor.jpg', edges)
    return edges

# Return color image (np.array([B,G,R]))
def projectImageColor(img1, img2, homography):
    # separate B, G and R into separate grey scale matrices
    BGR_img1 = cv2.split(img1)
    BGR_img2 = cv2.split(img2)
    result = []
    # get dimensions of base image
    w1, h1 = BGR_img1[0].shape
    # create mask and inverse mask
    mask = cv2.warpPerspective(np.ones(BGR_img2[0].shape), homography, dsize=(h1, w1),)
    maskInverted = np.ones(BGR_img1[0].shape) - mask
    # warp image to be projected
    for i in range(3):
        BGR_img2[i] = cv2.warpPerspective(BGR_img2[i], homography, dsize=(h1, w1))
        # create a blank area in the base image for each color
        BGR_img1[i] = np.asanyarray(np.multiply(BGR_img1[i], maskInverted), dtype='uint8')
        # Add base and projection image
        result.append(BGR_img1[i]+BGR_img2[i])
    return cv2.merge([result[0],result[1],result[2]])

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
def filterImage(image, actions = [1,2], kernel1 = np.ones((3,3), np.uint8), kernel2 = (5,5)):
    for action in actions:
        if action == 1:
            image = cv2.GaussianBlur(image, kernel2, 0.7)
            cv2.imwrite('Figures/LaTex_images/filter_corridor_Gaussian.jpg', image)
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

# return edge point in sub image [x,y]
def findEdgePointSubP(image, reverse=True, corner = 'bl'):
    h, w = image.shape
    # cornerthreshold is max # of pixels away from origin
    cornerthreshold = 30
    t = 200
    max_slope = 0
    edgepoint = [w,h]
    while edgepoint[0] == w and edgepoint[1] == h:
        if t < 50:
            break
        image = cv2.dilate(image,np.ones((2,2), np.uint8))
        hlines = cv2.HoughLinesP(image, rho=1, theta=1 * np.pi / 180, threshold=t, minLineLength=20, maxLineGap=20)
        if hlines is not None:
            for line in hlines:
                x1,y1,x2,y2 = line[0]
                if x1 != x2 and y1 < y2:
                    slope = (y2-y1)/(x2-x1)
                    if 0.2 < slope < 8:#7.5:
                        y_at_x0 = y1 - slope * x1
                        x_at_y0 = - y_at_x0/ slope
                        if (y_at_x0 < cornerthreshold or x_at_y0 < cornerthreshold) and slope > max_slope:
                            max_slope = slope
                            y_at_w = y_at_x0 + slope * w
                            x_at_h = (h-y_at_x0)/slope
                            if y_at_w > h:
                                if reverse:
                                    edgepoint = [w, y_at_w]
                                else:
                                    edgepoint = [x_at_h, h]
                            elif reverse:
                                edgepoint = [x_at_h, h]
                            else:
                                edgepoint = [w, y_at_w]
        t -= 20
    # save image
    mask = cv2.line(np.ones(image.shape),(int(edgepoint[0]),int(edgepoint[1])),(0,0),0,5)
    image = cv2.line(image,(int(edgepoint[0]),int(edgepoint[1])),(0,0),255,5)
    G = np.asanyarray(np.multiply(image, mask), dtype='uint8')
    R = np.asanyarray(np.multiply(image, mask), dtype='uint8')
    B = cv2.circle(image, (0, 0), 20, 0, thickness=-5)
    G = cv2.circle(G, (0, 0), 20, 0, thickness=-5)
    R = cv2.circle(R, (0, 0), 20, 255, thickness=-5)
    res = cv2.merge([B,G,R])
    cv2.imwrite('Figures/LaTex_images/filter_corner_'+corner+'.jpg',res)
    #cv2.imshow(corner, res)
    #cv2.waitKey(0)
    return edgepoint

# Return coordinates of vanishing points [vp_v, vp_h, vp_d]
def VanishingPointP(edgeImage):
       # Define upper and lower angle limits for houghline detection
    iterations = 0
    nrOfHlines = None
    t = 250
    hlines_horizontal = []
    hlines_vertical = []
    hlines_diagonal_up = []
    hlines_diagonal_down = []
    # Iterate until small sub set of lines if found
    while nrOfHlines is None or 30 > nrOfHlines or nrOfHlines > 60:
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
    diagImage = edgeImage.copy()
    for set in [hlines_diagonal_up, hlines_diagonal_down]:
        for line in set:
            diagImage = cv2.line(diagImage,(line[0],line[1]),(line[2],line[3]),123,thickness=10)
    cv2.imwrite('Figures/LaTex_images/filter_corner_diagonals.jpg',diagImage)
    cv2.waitKey(0)
    return [vp_v, vp_h, vp_d]

# return list of edge points [x,y] in the main image [tl,tr,bl,br]
def findAllEdgePointP(image, vp, reverse=True):
    vpX = int(vp[0])
    vpY = int(vp[1])
    for corner in ['tr','tl','br','bl']:
        if corner == 'bl':
            M = cv2.flip(image[vpY:, :vpX], 1)
            dxy = findEdgePointSubP(M, reverse, corner='bl')
            x = vpX - dxy[0]
            y = vpY + dxy[1]
            bl_pnt = [int(x), int(y)]
        elif corner == 'tr':
            M = cv2.flip(image[:vpY, vpX:], 0)
            dxy = findEdgePointSubP(M, reverse, corner='tr')
            x = vpX + dxy[0]
            y = vpY - dxy[1]
            tr_pnt = [int(x), int(y)]
        elif corner == 'tl':
            M = cv2.flip(cv2.flip(image[:vpY, :vpX], 0), 1)
            dxy = findEdgePointSubP(M, reverse, corner='tl')
            x = vpX - dxy[0]
            y = vpY - dxy[1]
            tl_pnt = [int(x), int(y)]
        elif corner == 'br':
            M = image[vpY:, vpX:]
            dxy = findEdgePointSubP(M, reverse, corner='br')
            x = vpX + dxy[0]
            y = vpY + dxy[1]
            br_pnt = [int(x), int(y)]
    return [tl_pnt, tr_pnt, bl_pnt, br_pnt]

# Return list with np.array of 4 pnt_dst [left,right,top,bottom]
def compute2DplaneCornerPoints(edgepoints, vpD):
    vp_x = vpD[0]
    vp_y = vpD[1]
    deltavp = 3
    cornerpoint_list = []
    tl_pnt, tr_pnt, bl_pnt, br_pnt = edgepoints
    for side in ['left','right','top','bottom']:
        if side == 'left':
            # LEFT WALL:
            pnt1 = tl_pnt
            pnt4 = bl_pnt
            pnt3 = [vp_x, vp_y + deltavp]
            pnt2 = [vp_x, vp_y - deltavp]
        elif side == 'right':
            # RIGHT WALL
            pnt2 = tr_pnt
            pnt3 = br_pnt
            pnt4 = [vp_x, vp_y + deltavp]
            pnt1 = [vp_x, vp_y - deltavp]
        elif side == 'top':
            # ceiling
            pnt1 = tl_pnt
            pnt2 = tr_pnt
            pnt3 = [vp_x + deltavp, vp_y]
            pnt4 = [vp_x - deltavp, vp_y]
        elif side == 'bottom':
            # floor
            pnt4 = bl_pnt
            pnt3 = br_pnt
            pnt1 = [vp_x - deltavp, vp_y]
            pnt2 = [vp_x + deltavp, vp_y]
        pts_dst = np.array([pnt1, pnt2, pnt3, pnt4])
        cornerpoint_list.append(pts_dst)
    return cornerpoint_list

# Return side ('left,right,top,bottom'
def determinePosition(vpD,edge_points,pixelposition):
    px = pixelposition[0]
    py = pixelposition[1]

    vpx = vpD[0]
    vpy = vpD[1]

    g1 = (vpy-py)/(vpx-px)

    tl_x = edge_points[0][0]
    tr_x = edge_points[1][0]
    bl_x = edge_points[2][0]
    br_x = edge_points[3][0]

    tl_y = edge_points[0][1]
    tr_y = edge_points[1][1]
    bl_y = edge_points[2][1]
    br_y = edge_points[3][1]

    if px < vpx and py < vpy:
        g2 = (vpy - tl_y) / (vpx - tl_x)
        if (g1 > g2):
            #print('top, left')
            pos = 'top'
        else:
            #print('left, top')
            pos = 'left'
    elif px < vpx and py > vpy:
        g2 = (vpy - bl_y) / (vpx - bl_x)
        if (g1 < g2):
            #print('bottom, left')
            pos = 'bottom'
        else:
            #print('left, bottom')
            pos = 'left'
    elif px > vpx and py < vpy:
        g2 = (tr_y - vpy) / (tr_x - vpx)
        if (g1 > g2):
            #print('right, top')
            pos = 'right'
        else:
            #print('top, right')
            pos = 'top'
    elif px > vpx and py > vpy:
        g2 = (br_y - vpy) / (br_x - vpx)
        if (g1 > g2):
            #print('bottom, right')
            pos = 'bottom'
        else:
            #print('right,bottom')
            pos = 'right'
    return pos

# Return image with vp and quadrant lines
def computeEdgeImages(vpD,edges):
    h0,w0 = edges.shape
    tempImage = cv2.merge([edges, edges, edges])
    tempImage = cv2.circle(tempImage, (int(vpD[0]), int(vpD[1])), 20, [0, 0, 255], thickness=-5)
    cv2.imwrite('Figures/LaTex_images/filter_corridor_VP.jpg', tempImage)

    mask = cv2.line(np.ones(edges.shape), (int(vpD[0]), 0), (int(vpD[0]), h0), 0, 10)
    mask = cv2.line(mask, (0, int(vpD[1])), (w0, int(vpD[1])), 0, 10)
    edgeslines = cv2.line(edges, (int(vpD[0]), 0), (int(vpD[0]), h0), 255, 10)
    edgeslines = cv2.line(edges, (0, int(vpD[1])), (w0, int(vpD[1])), 255, 10)

    Btemp = cv2.circle(edges, (int(vpD[0]), int(vpD[1])), 20, 0, thickness=-5)
    Gtemp = np.asanyarray(np.multiply(edgeslines, mask), dtype='uint8')
    Gtemp = cv2.circle(Gtemp, (int(vpD[0]), int(vpD[1])), 20, 0, thickness=-5)
    Rtemp = np.asanyarray(np.multiply(edgeslines, mask), dtype='uint8')
    Rtemp = cv2.circle(Rtemp, (int(vpD[0]), int(vpD[1])), 20, 255, thickness=-5)
    # cv2.imshow('Filter vp result', res)
    res = cv2.merge([Btemp, Gtemp, Rtemp])
    cv2.imwrite('Figures/LaTex_images/filter_corridor_lines.jpg', res)
    return res

# Return image with colored 2D planes
def project2DPlanes(originalGrey,cornerpoint_list):
    B = originalGrey
    G = originalGrey
    R = originalGrey
    h0,w0 = originalGrey.shape
    pts_src_grey = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]])
    for ii, side in enumerate(['left','right','top','bottom']):
        pts_dst = cornerpoint_list[ii]
        homography, status = cv2.findHomography(pts_src_grey, pts_dst)
        mask = np.ones(originalGrey.shape) - cv2.warpPerspective(np.ones(originalGrey.shape), homography,dsize=(w0, h0))
        if side == 'top':
            B = np.asanyarray(np.multiply(B, mask), dtype='uint8')
        elif side == 'bottom':
            G = np.asanyarray(np.multiply(G, mask), dtype='uint8')
        else:
            R = np.asanyarray(np.multiply(R, mask), dtype='uint8')
    res = cv2.merge([B,G,R])
    cv2.imwrite('Figures/LaTex_images/filter_corridor_results.jpg',res)
    return res

# Return dst_pnt
def findProjectionDstPnts(vpD,edgepoints,origin, h_length_m = 0.5, v_length_m = 1.2):
    side = determinePosition(vpD, edgepoints, origin)
    if side == 'left' or side == 'right':
        h_length = int(abs(vpD[0] - origin[0]) * h_length_m)
        v_length = h_length * v_length_m
        if origin[1] < vpD[1]:
            pnt1 = origin
            pnt4 = [origin[0], origin[1] + v_length]
        else:
            pnt1 = [origin[0], origin[1] - v_length]
            pnt4 = origin
        g12 = (vpD[1] - pnt1[1]) / (vpD[0] - pnt1[0])
        g43 = (vpD[1] - pnt4[1]) / (vpD[0] - pnt4[0])
        if side == 'left':
            pnt2 = [pnt1[0] + h_length, pnt1[1] + g12 * h_length]
            pnt3 = [pnt4[0] + h_length, pnt4[1] + g43 * h_length]
            pts_dst = np.array([pnt1, pnt2, pnt3, pnt4])
        else:
            pnt2 = [pnt1[0] - h_length, pnt1[1] - g12 * h_length]
            pnt3 = [pnt4[0] - h_length, pnt4[1] - g43 * h_length]
            pts_dst = np.array([pnt2,pnt1, pnt4,pnt3])
    else:
        v_length = int(abs(vpD[1] - origin[1]) * h_length_m)
        h_length = v_length * v_length_m
        if origin[0] < vpD[0]:
            pnt1 = origin
            pnt2 = [origin[0] + h_length, origin[1]]
        else:
            pnt1 = [origin[0] - h_length, origin[1]]
            pnt2 = origin
        g14 = (vpD[0] - pnt1[0]) / (vpD[1] - pnt1[1])
        g23 = (vpD[0] - pnt2[0]) / (vpD[1] - pnt2[1])
        if side == 'top':
            pnt4 = [pnt1[0] + g14 * v_length, pnt1[1] + v_length]
            pnt3 = [pnt2[0] + g23 * v_length, pnt2[1] + v_length]
            pts_dst = np.array([pnt1, pnt2, pnt3, pnt4])
        else:
            pnt4 = [pnt1[0] - g14 * v_length, pnt1[1] - v_length]
            pnt3 = [pnt2[0] - g23 * v_length, pnt2[1] - v_length]
            pts_dst = np.array([pnt4, pnt3, pnt2, pnt1])
    return pts_dst

# Return colorimage with projections
def projectImageWithMouse(original, vpD, edgepoints,floor_images,ceiling_images,wall_images):
    global mouseClick
    resultImage = original.copy()
    ii_w = 0
    ii_f = 0
    ii_c = 0
    cv2.imshow('result', resultImage)
    cv2.setMouseCallback('result',projectPoint)
    while(1):
        if mouseClick:
            origin= [x_coord,y_coord]#[100, 300]
            side = determinePosition(vpD, edgepoints, origin)
            if side == 'bottom':
                name = floor_images[ii_f]
                if ii_f == len(floor_images) - 1:
                    ii_f = 0
                else:
                    ii_f += 1
            elif side == 'top':
                name = ceiling_images[ii_c]
                if ii_c == len(ceiling_images) - 1:
                    ii_c = 0
                else:
                    ii_c += 1
            else:
                name = wall_images[ii_w]
                if ii_w == len(wall_images)-1:
                    ii_w = 0
                else:
                    ii_w += 1
            #for ii, name in names:
            #origin = origins[ii]
            projection = cv2.imread('Figures/' + name)
            h2,w2,BGR = projection.shape
            pts_src = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
            pts_dst = findProjectionDstPnts(vpD, edgepoints, origin,h_length_m=0.3,v_length_m=3)
            h, status = cv2.findHomography(pts_src, pts_dst)
            resultImage = projectImageColor(resultImage,projection,h)
            mouseClick = False
        cv2.imshow('result',resultImage)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('d'):
            break
        elif k == ord('r'):
            resultImage = original.copy()
    cv2.imwrite('Figures/LaTex_images/filter_corridor_results_AR.jpg', resultImage)
    cv2.destroyAllWindows()
    return resultImage

# Return colorimage with projections
def projectImageWithMouseQR(original, vpD, edgepoints,QR_codes):
    global mouseClick
    ii = 0
    lw_jj = 1
    rw_jj = 1
    c_jj = 1
    f_jj = 1
    resultImage = original.copy()
    w1, h1, RGB = resultImage.shape
    cv2.imshow('result', resultImage)
    cv2.setMouseCallback('result',projectPoint)
    while(1):
        if mouseClick:
            origin= [x_coord,y_coord]#[100, 300]
            side = determinePosition(vpD, edgepoints, origin)
            name = QR_codes[ii]
            if ii == len(QR_codes) - 1:
                ii = 0
            else:
                ii += 1
            qrcode = cv2.imread('Figures/' + name)
            qrcodeGrey = cv2.cvtColor(qrcode.copy(), cv2.COLOR_BGR2GRAY)
            h2,w2 = qrcodeGrey.shape
            pts_src = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
            pts_dst = findProjectionDstPnts(vpD, edgepoints, origin,h_length_m=0.3,v_length_m=3)
            x1 = min([int(x[0]) for x in pts_dst])
            x2 = max([int(x[0]) for x in pts_dst])
            y1 = min([int(x[1]) for x in pts_dst])
            y2 = max([int(x[1]) for x in pts_dst])
            h, status = cv2.findHomography(pts_src, pts_dst)
            resultImage = projectImageColor(resultImage,qrcode,h)
            # create white background
            mask = 255*(np.ones((w1,h1)) - cv2.warpPerspective(np.ones(qrcodeGrey.shape), h, dsize=(h1, w1)))
            qrcodewarped = np.asanyarray(mask+ cv2.warpPerspective(qrcodeGrey, h, dsize=(h1, w1)), dtype='uint8')
            qrcodeResult = np.array(qrcodewarped)[y1:y2, x1:x2]
            if side == 'left':
                nr = str(lw_jj)
                lw_jj += 1
            if side == 'right':
                nr = str(rw_jj)
                rw_jj += 1
            if side == 'bottom':
                nr = str(f_jj)
                f_jj += 1
            if side == 'top':
                nr = str(c_jj)
                c_jj += 1
            cv2.imwrite('Figures/LaTex_images/QR_' + side + '_'+nr + '.jpg', qrcodeResult)
            mouseClick = False
        cv2.imshow('result',resultImage)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('d'):
            break
        elif k == ord('r'):
            resultImage = original.copy()
            lw_jj = 1
            rw_jj = 1
            c_jj = 1
            f_jj = 1
    cv2.imwrite('Figures/LaTex_images/Projected_QR_codes.jpg', resultImage)
    cv2.destroyAllWindows()
    return resultImage,h

def mainFunction(imageName = 'Figures/corridor.jpg'):
    # 1. Read image and convert to grayscale
    original = cv2.imread(imageName)
    originalGrey = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)

    # 2. compute filtered image (canny)
    edges = filterImage(originalGrey)

    # 3.  vanishing points from filtered image
    vp_all = VanishingPointP(edges)
    if vp_all is not None:
        vpV = vp_all[0]
        vpH = vp_all[1]
        vpD = vp_all[2]
        if vpH is None and vpD is None:
            return originalGrey
    else:
        return originalGrey

    # 4. Find edge points
    edgepoints = findAllEdgePointP(edges,vpD,reverse=False)

    # 5. Find cornerpoints of 2D planes
    cornerpoint_list = compute2DplaneCornerPoints(edgepoints,vpD)

    # 6. Compute grey images of VP and quadrants
    computeEdgeImages(vpD, edges)

    # 7. Project 2D planes on top of grey source image
    project2DPlanes(originalGrey, cornerpoint_list)

    # 8. Project images on original base image
    wall_images = ['fakelove.jpg','gta.jpg','monalisa2.jpg','banksy2.jpg']
    floor_images = ['carpet.jpg']
    ceiling_images = ['sky.jpg']
    resultImage = projectImageWithMouse(original, vpD, edgepoints, floor_images, ceiling_images, wall_images)

    # 9. Project QR_codes on original base image and save QR code
    QR_codes = ['QR_real.png']
    resultImage, homographyQR = projectImageWithMouseQR(original, vpD, edgepoints, QR_codes)

    return resultImage, homographyQR

def getWarpedTemplate(original):#'Figures/corridor.jpg'):
    # 1. Read image and convert to grayscale
    #original = cv2.imread(imageName)
    originalGrey = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)

    # 2. compute filtered image (canny)
    edges = filterImage(originalGrey)

    # 3.  vanishing points from filtered image
    vp_all = VanishingPointP(edges)
    if vp_all is not None:
        vpV = vp_all[0]
        vpH = vp_all[1]
        vpD = vp_all[2]
        if vpH is None and vpD is None:
            return originalGrey
    else:
        return originalGrey

    # 4. Find edge points
    edgepoints = findAllEdgePointP(edges,vpD,reverse=False)

    # 5. Find cornerpoints of 2D planes
    cornerpoint_list = compute2DplaneCornerPoints(edgepoints,vpD)

    # 6. Compute grey images of VP and quadrants
    computeEdgeImages(vpD, edges)

    # 7. Project 2D planes on top of grey source image
    project2DPlanes(originalGrey, cornerpoint_list)

    # 8. Project images on original base image
#    wall_images = ['fakelove.jpg','gta.jpg','monalisa2.jpg','banksy2.jpg']
#    floor_images = ['carpet.jpg']
#    ceiling_images = ['sky.jpg']
#    resultImage = projectImageWithMouse(original, vpD, edgepoints, floor_images, ceiling_images, wall_images)

    # 9. Project QR_codes on original base image and save QR code
    QR_codes = ['QR_real.png']
    resultImage, homographyQR = projectImageWithMouseQR(original, vpD, edgepoints, QR_codes)

    return resultImage, homographyQR

def main():
    imageList = [
        'KjellerHallway_with_QR'
        #'hallway',
        #'hallway2',
        #'hallway3',
        #'corridor',
        #'corridor1',
        #'corridor4',
    ]

    for name in imageList:
        image_result, h = mainFunction('Figures/'+name+'.jpg')
    cv2.destroyAllWindows()

if __name__ == "__main__": main()