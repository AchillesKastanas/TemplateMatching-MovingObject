import cv2
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
from os import listdir 
import glob

def  aspect_ratio_resize(image,  width=None,  height=None,  inter=cv2.INTER_AREA):
   dim =  None
   (h, w)  = image.shape[:2]
   if width is  None  and height is  None:
     return image

   if width is  None:
      r = height /  float(h)
      dim =  (int(w * r), height)
   else:
      r = width /  float(w)
      dim =  (width,  int(h * r))
   return cv2.resize(image, dim,  interpolation=inter)

#Load Template convert to grayscale and perform canny edge detection
template = cv2.imread('template.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template,  50,  200)
(tH, tW)  = template.shape[:2]

#split video to frames and store them
vidcap = cv2.VideoCapture('ballvideo.mp4')
success,image = vidcap.read()
count = 0
while success:
   cv2.imwrite("frames/frame%d.jpg" % count, image)     
   success,image = vidcap.read()
   count += 1

#For every single frame of the video
for i in range(len(os.listdir('./frames'))):
   path = r'frames/frame%d.jpg' % i
   print(path)

   original_image = cv2.imread(path)
   final = original_image.copy()
   gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
   found =  None

   for scale in np.linspace(0.2,  1.0,  20)[::-1]:
        resized = aspect_ratio_resize(gray,  width=int(gray.shape[1]  * scale))
        r = gray.shape[1]  /  float(resized.shape[1])

        if resized.shape[0]  < tH or resized.shape[1]  < tW:
           break
        canny = cv2.Canny(resized,  50,  200)
        detected = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF)
        (_, max_val, _, max_loc)  = cv2.minMaxLoc(detected)

        if found is  None  or max_val > found[0]:
           found =  (max_val, max_loc, r)

   (_, max_loc, r)  = found
   (start_x, start_y)  =  (int(max_loc[0]  * r),  int(max_loc[1]  * r))
   (end_x, end_y)  =  (int((max_loc[0]  + tW)  * r),  int((max_loc[1]  + tH)  * r))

   #fill background accordingly
   tempimage = PIL.Image.open(path)
   rgb_tempimage = tempimage.convert("RGB")
   backgroundcolor = rgb_tempimage.getpixel((start_x, start_y))
   bgr_backgroundcolor = (backgroundcolor[2], backgroundcolor[1], backgroundcolor[0])

   cv2.rectangle(final,  (start_x, start_y),  (end_x-10, end_y),  bgr_backgroundcolor,  -1)
   cv2.imwrite("finalframes/frame%d.jpg" % i, final)
   cv2.waitKey(0)

#export frames to video
img_array = []
for i in range(len(os.listdir('./finalframes'))):
   path = r'finalframes/frame%d.jpg' % i
   img = cv2.imread(path)
   height, width, layers = img.shape
   size = (width,height)
   img_array.append(img)

out = cv2.VideoWriter('finalvideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
   out.write(img_array[i])
out.release()