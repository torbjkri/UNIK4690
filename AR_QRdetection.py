# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:38:28 2018

@author: wubst
"""# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:22:28 2018

@author: wubst
"""

import cv2
import numpy as np
import QR_detectionFunctions as wfc



class Image:
    def __init__(self,image):
        self.image = image#cv2.imread(file)
        
class Template:
    def __init__(self,image,transformed = False,homography = None):
            self.image = image
            self.transform = homography
            self.transformed = transformed

def matchingInit():
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matchDetector = cv2.FlannBasedMatcher(index_params, search_params)
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 4)
    return matchDetector, sift

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

def projectAndTransform(scene,image,homography):
    thresh=15
    maxval = 255
        
    w1, h1 = scene.shape[:2]
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    image_warped = cv2.warpPerspective(image, homography, dsize=(h1, w1))

    image_gray_warped = cv2.warpPerspective(image_gray,homography, dsize=(h1,w1))
    ret, mask = cv2.threshold(image_gray_warped, thresh, maxval, cv2.THRESH_BINARY)  
    #kernel = np.ones((20,20), np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    inv = cv2.bitwise_not(mask)
    
    scene_bg = cv2.bitwise_and(scene,scene,mask = inv)
    image_fg = cv2.bitwise_and(image_warped,image_warped, mask = mask)
    result = cv2.add(scene_bg, image_fg)

            

    return result, image_fg

def rotateMatrix(theta,image):

    
    angle = (-theta)*np.pi/180
    w,h, d = image.shape
    
    A1 = np.array([[1,0,-w/2], \
                   [0,1,-h/2], \
                   [0,0,0], \
                   [0,0,1.0]])
    
    rmatrix = np.array([[1,0,0,0], \
                   [0,np.cos(angle),-np.sin(angle),0], \
                   [0,np.sin(angle),np.cos(angle),0], \
                   [0,0,0,1.0]])
    
    T = np.array([[1,0,0,0],\
                  [0,1,0,0],\
                  [0,0,1,450],\
                  [0,0,0,1]])
    
    A2 = np.array([[200,0,w/2,0], \
                   [0,200,np.sin(40*np.pi/180)*h/2,0], \
                   [0,0,1,0]]) \
                   
                   
    transform = A2 @ ( T@ (rmatrix@ A1))
    
    img2 = cv2.warpPerspective(image, transform,(h,int(np.cos(40*np.pi/180)*w)), cv2.INTER_LANCZOS4)
    
    return transform, img2






def mainFunction(QR_files, painting_files, testing = True, moving = False, template_transforming = False, size_change = False, pre_transform = False, manual_transform = False):
    
    #testing = False
    
    painting_size = [[50.,50.],[50.,50.],[50.,50.]]
    template_size = [[18.,18.],[18.,13.5],[21.,13.5]]
    
    
    matchDetector, sift = matchingInit()
    translation = np.array([[1.,0.,0.],[0.,1.,0.]])
    MIN_MATCH_COUNT = 15
    
    if testing == False:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1920)
    else:
        print('HEY')
        screenshot = cv2.imread('Figures/KjellerHallway_with_QR.jpg')
        scene_target = screenshot
        ret = True
    
    
    templates = []
    paintings = []
    paintings_gray = []
    des_template = []
    kp_template = []
    
    
    
    
    cv2.namedWindow('template',cv2.WINDOW_NORMAL)
    
    for ii, template in enumerate(QR_files):    
        temp = cv2.imread(template)
        if pre_transform:
            if manual_transform:
               homography, _ = rotateMatrix(60,temp)
            else:
                _, homography = wfc.getWarpedTemplate(screenshot)
            w,h = screenshot.shape[:2]
            temp = cv2.warpPerspective(temp,homography, dsize =(h,w))
            image = Template(temp,True,homography)
        else:
            image = Template(temp,False,None)
        kp, des = sift.detectAndCompute(image.image,None)
        kp_template.append(kp)
        des_template.append(des)   
        
        image.image = cv2.cvtColor(image.image.copy(),cv2.COLOR_BGR2GRAY)
        templates.append(image)
        
        
    for ii, painting in enumerate(painting_files):
        temp = cv2.imread(painting)
        
        if size_change:
            ht, wt = templates[ii].image.shape[:2]
            w = int(wt*painting_size[ii][0]/template_size[ii][0])
            h = int(ht*painting_size[ii][1]/template_size[ii][1])
            temp = cv2.resize(temp, dsize = (w,h), interpolation = cv2.INTER_CUBIC)
            
        elif pre_transform:
            w,h = screenshot.shape[:2]
            temp = cv2.warpPerspective(temp,templates[ii].transform, dsize =(h,w))
        
        image = Image(temp)
        paintings.append(image)
        image2 = cv2.cvtColor(image.image.copy(),cv2.COLOR_BGR2GRAY)
        paintings_gray.append(image2)
    
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    
    counter = 0
    
    
    while(1):
        
        if testing == False:
            ret, scene_target = cap.read()
    
        if ret:
            scene_gray = cv2.cvtColor(scene_target, cv2.COLOR_BGR2GRAY)
            kp_target, des_target = sift.detectAndCompute(scene_gray,None)
            for ii, template in enumerate(templates):
                #found_match = False
                goodMatches = findGoodMatches(des_template[ii], des_target, matchDetector)
                
                if len(goodMatches) > MIN_MATCH_COUNT:
                    
                    homography = findHomography(kp_template[ii], kp_target, goodMatches)
                    if template_transforming:
                        template.homography = homography
                        w,h = scene_target.shape[:2]
                        template.image = cv2.warpPerspective(template.image,homography,dsize = (h,w))
                        kp_template[ii], des_template[ii] = sift.detectAndCompute(template.image,None)
        
                        scene_target, paintings[ii].image = projectAndTransform(scene_target, paintings[ii].image, homography)
                        #cv2.imwrite('Figures/ART_update{}.jpg'.format(counter),scene_target)
                        counter += 1
                    elif moving:
                        
                        w,h = paintings[ii].image.shape[:2]
                        h += int(abs(translation[0,2]))
                        w += int(abs(translation[1,2]))
                        temp = cv2.warpAffine(paintings[ii].image, translation, dsize = (h,w))
                        
                        scene_target, _ = projectAndTransform(scene_target, temp, homography)
                    else:
                        scene_target, _ = projectAndTransform(scene_target, paintings[ii].image, homography)
                    
                elif template_transforming:
                    
                    paintings[ii].image = cv2.imread(painting_files[ii])
                    template.image = cv2.imread(QR_files[ii])
                    template.homography = None
                    kp_template[ii], des_template[ii] = sift.detectAndCompute(template.image,None)
                    
            
            counter += 1
            cv2.imshow('result', scene_target)
        
                    
          
        else:
            print('Camera Fuckup')
            
        if testing:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
            break
            
        
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
        elif k == 27:
            cv2.imwrite('Figures/QR_hallway_detection.jpg',scene_target)
        elif k == ord('a'):
            translation[0,2] -= 50
        elif k == ord('d'):
            translation[0,2] += 50
        elif k == ord('a'):
            translation[1,2] -= 50
        elif k == ord('d'):
            translation[1,2] += 50
            

def main():

    
    #QR_files = ['Figures/QR_real.png','Figures/qrwall2.png','Figures/qrwall1.png']
    #painting_files = ['Figures/fakelove.jpg','Figures/monalisa2.jpg','Figures/gtalady2.jpg']
    
    QR_files = ['Figures/QR_real.png']#,'Figures/qrwall1.png']
    painting_files = ['Figures/banksy2.jpg']#,'Figures/banksy2.jpg']
    
    mainFunction(QR_files, painting_files, testing = False, moving = False, template_transforming = False, size_change = False)
    
#    mainFunction(QR_files, painting_files, testing = True, moving = False, template_transforming = False, size_change = False, pre_transform = True, manual_transform = False)

if __name__ == "__main__": main()