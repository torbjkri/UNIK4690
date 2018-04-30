import cv2
import numpy as np

def projectImage(img1, img2, homography, thresh = 10, maxval = 255):

    w2, h2 = img1.shape
    img2_warped = cv2.warpPerspective(img2, homography, dsize=(h2, w2))
    #cv2.imshow('before',img2)
    #cv2.imshow('after',img2_warped)
    #cv2.waitKey(0)
    ret, mask = cv2.threshold(img2_warped, thresh, maxval, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1, inv)
    img2_fg = cv2.bitwise_and(img2_warped, mask)

    return cv2.add(img1_bg, img2_fg)


# Find good matches between images, and return them
def findGoodMatches(des1, des2):
    # FLANN: Fast Library for Approximate Nearest Neighbour
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matchDetector = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matchDetector.knnMatch(des1, des2, k = 2)
    goodMatches = []
    
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)
            
    return goodMatches


##Given keypoints in two images and good matches between them, compute
## homography using RANSAC
def findHomography(kp1, kp2, goodMatches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)
    
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return homography