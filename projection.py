import cv2
import numpy as np

def projectImage(img1, img2, homography, thresh, maxval, transform, transformed):
    
    w2, h2, d2 = img2.shape
    
    if transformed:
        img2 = cv2.warpPerspective(img2,transform,(h2,w2))
    
    w2, h2, d2 = img1.shape
    #w2, h2, d2 = img1.shape
    img2_warped = cv2.warpPerspective(img2, homography, dsize=(h2, w2))
    #cv2.imshow('before',img2)
    #cv2.imshow('after',img2_warped)
    #cv2.waitKey(0)
    
    img2_warped_gray = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2_warped_gray, thresh, maxval, cv2.THRESH_BINARY)
  
    kernel = np.ones((20,20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
#    print(type(mask))
#    print(img1.shape)
#    print(img2_warped.shape)
#    print(mask.shape)
    
    inv = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(img1,img1,mask = inv)
    img2_fg = cv2.bitwise_and(img2_warped,img2_warped, mask = mask)
#    temp = cv2.add(mask,inv)
#    cv2.imshow('masks',temp)
#    cv2.imshow('mask',mask)
#    cv2.imshow('inv',inv)
#    cv2.imshow('warped',img2_warped)
#    cv2.imshow('warped gray',img2_warped_gray)
#    cv2.imshow('bg', img1_bg)
#    cv2.imshow('fg',img2_fg)
#    cv2.waitKey(0)
    result = cv2.add(img1_bg, img2_fg)
    
    
    return result


# Find good matches between images, and return them
def findGoodMatches(des1, des2, matchDetector):
    # FLANN: Fast Library for Approximate Nearest Neighbour
    

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

def rotateMatrix(theta,image):

    
    angle = (-theta)*np.pi/180
    w,h, d = image.shape
    
    A1 = np.array([[1,0,-w/2], \
                   [0,1,-h], \
                   [0,0,0], \
                   [0,0,1.0]])
    
    rmatrix = np.array([[1,0,0,0], \
                   [0,np.cos(angle),-np.sin(angle),0], \
                   [0,np.sin(angle),np.cos(angle),0], \
                   [0,0,0,1.0]])
    
    T = np.array([[1,0,0,0],\
                  [0,1,0,0],\
                  [0,0,1,300],\
                  [0,0,0,1]])
    
    A2 = np.array([[200,0,w/2,0], \
                   [0,200,h/2,0], \
                   [0,0,1.0,0]]) \
                   
                   
    transform = A2 @ ( T@ (rmatrix@ A1))
    
    img2 = cv2.warpPerspective(image, transform,(h,int(w/2)), cv2.INTER_LANCZOS4)
    
    return transform, img2
