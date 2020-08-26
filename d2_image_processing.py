# Image Processing before CNN training
import cv2
import numpy as np
import os
import glob
np.warnings.filterwarnings('ignore')
# NMS slow ver.
def non_max_suppression_slow(boxes, overlapThresh=0.5):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# initialize the list of picked indexes
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
	return boxes[pick]


# Malisiewicz et al. 
# NMS
def non_max_suppression_fast(boxes, overlapThresh=0.5):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

idx = 0
path = 'F:/semigradpro/pic_pre/*'

images = glob.glob(path)
for fname in images:
	print(fname+'\n')
	keep = []
	boxes = []
	vec = []
	threshold = 100
	minLength = 80
	lineGap = 5
	rho = 1
	limit = 150
	# load image
	img = cv2.imread(fname,cv2.IMREAD_COLOR)
	try:
		height, width = img.shape[:2]
	except:
		print('no shape')
		break
	# img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
	#graysacle
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
	
    # Thresholding
    # gray_pin = 196
    # ret, thresh = cv2.threshold(denoised, gray_pin, 255, cv2.THRESH_BINARY)

	kernel = np.ones((2,2), np.uint8)
	close_kernel = np.ones((9, 5), np.uint8)
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	mean_thres = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 12)
	close = cv2.morphologyEx(mean_thres,cv2.MORPH_CLOSE,close_kernel)
	lines = cv2.HoughLinesP(close,rho,np.pi/180,threshold,minLength, lineGap)
	try:
		if(len(lines)>0):
			for i in range(0,len(lines)):
				vec = lines[i][0]
				pt1 = (vec[0],vec[1])
				pt2 = (vec[2],vec[3])
				gapY = abs(vec[3]-vec[1])
				gapX = abs(vec[2]-vec[0])
				if(gapY>limit and limit >0):
					cv2.line(close, pt1, pt2, (0,0,0), 10)
	except:
		print('no lines')

	contours,hierarchy = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		boxes.append([x, y, w, h])
	boxes = np.array(boxes)  
	keep = non_max_suppression_fast(boxes)
	# keep = non_max_suppression_slow(boxes)
	# print(len(boxes))
	# print(len(keep))

	# for b in boxes:
	# 	x,y,w,h = b
	# 	if (w>40 and h>10) and (w < width/2 and h < height/2):
	# 		img1 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	idx2 = 0
	for k in keep:
		x,y,w,h = k
		if w>40 and h>10 and w < width/2 and h < height/2:
			img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			idx+=1            
			idx2+=1                       
			new_img = img[y:y+h,x:x+w]
			cv2.imwrite('./preprocessing/'+str(idx) + '.png', new_img)
	cv2.imwrite('./preprocessing/whole'+str(idx2)+'.png',img)
	
	# check contour image boxes
	# cv2.imshow('1',img1)
	# cv2.imshow('2',img2)
	# try:
	# 	cv2.imwrite('./temp_pre/'+str(idx)+'.png',img2)
	# 	idx+=1
	# except:
	# 	print('no rectangle')
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     if w>50 and h>50 and w < width/2 and h < height/2:
    #         idx+=1
    #         new_img=gradient[y:y+h,x:x+w]
    #         cv2.imwrite('./preprocessing/'+str(idx) + '.png', new_img)
print('Done')                                      

