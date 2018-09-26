#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  GetMark.py
#  
#  Copyright 2018 User <User@DESKTOP-17Q7VC8>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import math
import imutils
import time
from pandas import DataFrame
def doMorphologyEx(im,method,kern):
	out = cv2.morphologyEx(im, method, kernel)
	return out
	
def doAdaptiveThreshold(image):
	out = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
	return out

def doGaussianBlur(im,numhere):
	out = cv2.GaussianBlur(im,numhere ,0)
	return out
	
def doMedianBlur(im,numhere):
	out = cv2.medianBlur(im,numhere)
	return out

def doBlur(im,numhere):
	out = cv2.blur(im,numhere)
	return out

def doThreshold(im):
	out = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
	return out

def resizeToFit(img):
	# [GET] region of every questions section
	height,width = img.shape[:2]
	height = int((height/4)*1)
	width = int((width/4)*1)
	resized_img = cv2.resize(img, (width, height)) 
	
	return resized_img	

def divideSection(bubbleregion):

	# [GET] region of every questions section
	height,width = bubbleregion.shape[:2]
	print("height", height)
	print("width", width)

	Y = height
	WidthStart = 0
	sect = list()
	for i in range(2):
		WidthEnd = int(width * (i+1)/2)
		print(i)
		if i == 0:
			WidthStart = WidthStart + 10
			
		if i == 1:
			WidthStart = WidthStart - 5
			WidthEnd   = WidthEnd 
			
		
		sect.insert(i,(bubbleregion[0:Y , WidthStart+30:WidthEnd]))
		WidthStart = WidthEnd
	return sect

def findAllCnts(image):
	
	kernel = np.ones((3, 3), np.uint8)
	# [IMAGE PROCESSING] morphology opening > convert color > doGaussianBlur > doAdaptiveThreshold > do morphology closing
	image = resizeToFit(image)
	cv2.imshow("test1",image)

	graycrop = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	cv2.imshow("test2",graycrop)
	#blur1 =  cv2.medianBlur(graycrop,5)
	blur1 = doGaussianBlur(graycrop,(5,5))
	cv2.imshow("test3",blur1)
	#test	ret, thresh1 = cv2.threshold(blur1, 210, 255, cv2.THRESH_BINARY)
	#thresh1 = cv2.threshold(blur1, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh1 = doAdaptiveThreshold(blur1)
	cv2.imshow("test4",thresh1)
	# done image process

	# [FIND] find contours here after process
	_,cnts,_ = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	
	docCnt = None
	print(len(cnts))
	# ensure that at least one contour was found
	return cnts

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
	#pyimagesearch
def findBubble(img):
	
	# [FIND] find contours here after process
	#edit sini sakni kul 3:55 kat bubbleregion ganti sect[i]
	kernel = np.ones((3, 3), np.uint8) #3
	img = doMorphologyEx(img, cv2.MORPH_OPEN, kernel)
	brcrop = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#brblur =  cv2.medianBlur(brcrop,1)  #1
	brblur = doGaussianBlur(brcrop,(3,3))#1
	
	brthresh = doAdaptiveThreshold(brblur)
	kernel = np.ones((1, 1), np.uint8)
	bthresh = doMorphologyEx(brthresh, cv2.MORPH_CLOSE, kernel)
	_,brcnts,_ = cv2.findContours(brthresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	# done find contours

	# [FILTER] filter the bubble from other contours
	print("brcnts length ",len(brcnts))

	newbrcnts = []
	for c in brcnts:
		area = cv2.contourArea(c)
		
		if area > 120 and area < 300:
			perimeter = cv2.arcLength(c,True)

			if perimeter < 100 and perimeter > 45:
				
				newbrcnts.append(c)
	# done process
	bubblecnts = []
	 
	# loop over the contours
	for c in newbrcnts:
		# compute the bounding box of the contour, then use the
		# bounding box to derive the aspect ratio
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
	 
		# in order to label the contour as a question, region
		# should be sufficiently wide, sufficiently tall, and
		# have an aspect ratio approximately equal to 1
		if w >= 7 and h >= 7 and ar >= 0.1 and ar <= 1.2:
			bubblecnts.append(c)

	# [SORT] sort contours 1	
	##bubblecnts = contours.sort_contours(bubblecnts,method="top-to-bottom")[0]
	# [SORT] sort contours 2
	bubblecnts.sort(key=lambda x:get_contour_precedence(x, sect[1].shape[0]))    
	# [DONE] done contours extract and sort
		
	
	return bubblecnts
#a6, a11 tolerant 19

def get_contour_precedence(cnt, cols):
    tolerance_factor = 15 #a6.jpg use 23 19 #a7 use 3 cameraloo =15 20
    origin = cv2.boundingRect(cnt)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

kernel = np.ones((3, 3), np.uint8)	
image = cv2.imread("a6.jpg") #a6,a7,a11 
image = resizeToFit(image)
cv2.imshow("image",image)
sect  = divideSection(image)



for i in range(2):
	cnts = findBubble(sect[i])
cnts2 = findBubble(sect[0])
x = -1
for i in range(15):

	cv2.drawContours(sect[0],cnts2,x, (0,0,255), 0)
	cv2.drawContours(sect[1],cnts,x, (0,0,255), 0)
	x = x + 5
cv2.imshow("test0",sect[0])
cv2.imshow("test1",sect[1])
cv2.waitKey(0)
