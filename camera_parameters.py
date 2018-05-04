# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:04:40 2018

@author: wubst
"""

import cv2


def camparam():
    K = np.array([[649.25839464 ,  0.   ,      323.60818508], /
                  [  0.         ,651.0548515  ,242.27633613], /
                  [  0.           ,0.           ,1.        ]])
    dist = np.array([[-0.0050787   ,0.22863201 ,-0.00224043  ,0.00420507 ,-0.46658716]])
    
    return K, dist