# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:22:28 2018

@author: wubst
"""

import cv2
import numpy as np
import projection as pr



class Image:
    def __init__(self,file):
        self.image = cv2.imread(file)
        
class Template:
    def __init__(self,file,rotated):
        if rotated: 
            temp = cv2.imread(file)
            self.image = temp.copy()
            self.transform, _ = pr.rotateMatrix(60,temp)
            self.transformed = True
        else:
            self.image = cv2.imread(file)
            self.transform = None
            self.transformed = False
        
            
        
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matchDetector = cv2.FlannBasedMatcher(index_params, search_params)
#matchDetector = cv2.BFMatcher(cv2.NORM_HAMMING2)

QR_files = ['Figures/QR_real.png','Figures/table.png']
painting_files = ['Figures/fakelove.jpg','Figures/monalisa.jpg','Figures/gtaldy.jpg']
#painting_files = ['Figures/banksy.jpg','Figures/fakelove.jpg']

MIN_MATCH_COUNT = 10

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1920)

sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 3)
#sift = cv2.ORB_create()
#sift = cv2.xfeatures2d.SURF_create(400,5,5)

templates = []
paintings = []
paintings_gray = []

screenshot = cv2.imread('Figures/.jpg')

testing = False

des_template = []
kp_template = []

for ii, template in enumerate(QR_files):    
    
    if ii == 0:
        image = Template(template,True)
    else:
        image = Template(template,False)
    kp, des = sift.detectAndCompute(image.image,None)
    kp_template.append(kp)
    des_template.append(des)
    cv2.imshow(str(ii), image.image)    
    
    image.image = cv2.cvtColor(image.image.copy(),cv2.COLOR_BGR2GRAY)
    templates.append(image)
    
for painting in painting_files:
    image = Image(painting)
    paintings.append(image)
    image2 = cv2.cvtColor(image.image.copy(),cv2.COLOR_BGR2GRAY)
    paintings_gray.append(image2)

cv2.namedWindow('result',cv2.WINDOW_NORMAL)

while(1):
    
    if testing == True:
        ret = True
        scene_target = screenshot
    else:
        ret, scene_target = cap.read()
    
    
    if ret:
        scene_gray = cv2.cvtColor(scene_target, cv2.COLOR_BGR2GRAY)
        
        
        
        
        
        for ii, template in enumerate(templates):
            #found_match = False
            if template.transformed:
                transform_scene,temp = pr.rotateMatrix(-60,scene_target)
                #temp = cv2.warpPerspective(scene_gray,np.linalg.inv(template.transform), (scene_gray.shape[1],scene_gray.shape[0]))
                kp_target, des_target = sift.detectAndCompute(temp,None)
            
                goodMatches = pr.findGoodMatches(des_template[ii], des_target, matchDetector)
                cv2.imshow('temp',temp)
                if len(goodMatches) > MIN_MATCH_COUNT:
                    homography = pr.findHomography(kp_template[ii], kp_target, goodMatches)
                    scene_target = pr.projectImage(temp, paintings[ii].image, homography,15,255,transform_scene,template.transformed)
            else:
                kp_target, des_target = sift.detectAndCompute(temp,None)
            
                goodMatches = pr.findGoodMatches(des_template[ii], des_target, matchDetector)
                
                if len(goodMatches) > MIN_MATCH_COUNT:
                    homography = pr.findHomography(kp_template[ii], kp_target, goodMatches)
                    scene_target = pr.projectImage(scene_target, paintings[ii].image, homography,15,255,template.transform,template.transformed)
                    
        cv2.imshow('result', scene_target)
        
                
        
    else:
        print('Camera Fuckup')
        
    
    k = cv2.waitKey(10) & 0xFF
    if k == 32:
        cv2.destroyAllWindows()
        cap.release()
        break
        