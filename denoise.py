#!/usr/bin/env python

#@ ---title--- image denoising with python
#@ ---abstract--- remove text background with cv2 and numpy 
#@  

#@ I'm a big fan of the <a id = "NCE" href = https://www.camscanner.com/>CamScanner</a> app 
#@ and have been using this program on my phone for many years, ever since a friend introduced me to it. 
#@ You take a snapshot of a piece of paper and *CamScanner* will remove the background and produce
#@ a pdf with crisp, very readable, text. I wanted to add this functionality to my <a id = "NCE" href = https://github.com/kacpertopol/cam_board> 
#@ cam_board</a> script and after some experimentation I think I have a simple solution.  
#@ Below is a description of the resulting procedure. 
#@  

#@ First some imports 
#@ref imports 
#@insert imports
#@ The `argparse` library will be used to parse the command line arguments. This might be  
#@ overkill since there will be only one argument - the name of the image (without the extension) 
#@ that will undergo the denoising procedure. The `cv2` in conjunction with the `numpy` library 
#@ will be used to perform image manipulations.

#@begin imports
import argparse
import cv2
import numpy
#@end imports


parser = argparse.ArgumentParser(description = "Denoise image.")
parser.add_argument("input" , help = "Input image name (without extension)")
args = parser.parse_args() 

warped = cv2.imread(args.input + ".png")

gray = cv2.cvtColor(warped , cv2.COLOR_BGR2GRAY) 
gray = 255 - gray

cv2.imwrite(args.input + "_gray.png" , gray)

gray = gray.astype("float32")

blur_kernel = numpy.ones((300 , 300) , dtype = numpy.float32) 
blur_kernel = blur_kernel / numpy.sum(blur_kernel.flatten())

blured_gray = cv2.filter2D(gray , -1 , blur_kernel)

cv2.imwrite(args.input + "_blured_1.png" , blured_gray)

gray[0:3 , :] = 0.0
gray[gray.shape[0] - 3 : gray.shape[0] , :] = 0.0
gray[: , 0:3] = 0.0
gray[: , gray.shape[1] - 3 : gray.shape[1]] = 0.0

blured_gray = cv2.filter2D(gray , -1 , blur_kernel)

cv2.imwrite(args.input + "_blured_2.png" , blured_gray)

stdv = numpy.sqrt(numpy.mean(((gray - blured_gray) * (gray - blured_gray)).flatten()))

gray_2 = (gray - blured_gray)
gray_2 = 255.0 * (gray_2 - numpy.amin(gray_2)) / (numpy.amax(gray_2) - numpy.amin(gray_2))
cv2.imwrite(args.input + "_gray_2.png" , gray_2)

hls = cv2.cvtColor(warped , cv2.COLOR_BGR2HLS)

h_res = hls[: , : , 0]
l_res = numpy.full(gray.shape , 255.0 , dtype = numpy.float32)
l_res = numpy.where((gray - blured_gray) > stdv , hls[: , : , 1] , l_res)
s_res = hls[: , : , 2]

warped = cv2.cvtColor(cv2.merge((h_res.astype("uint8") , l_res.astype("uint8") , s_res.astype("uint8"))) , cv2.COLOR_HLS2BGR)

cv2.imwrite(args.input + "_warped.png" , warped)

