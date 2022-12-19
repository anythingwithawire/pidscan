#!/bin/python3
#-t tp -i z.tif -c blue -g 0.45,0.45,0.45 -s square -p 1 -r 1 -mm 60,50,30,45 -ll 10,-10,65,85 -d 0 -z kkk.tif  mid left
# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
from imutils.object_detection import non_max_suppression
import os


def fatten(tifImage):
	# improve line thickness
	_, thresh = cv2.threshold(tifImage, 250, 255, cv2.THRESH_BINARY)
	# dilated = cv2.dilate(thresh, np.ones((3, 3)))
	kernel = np.ones((1, 1), 'uint8')
	dilate_img = cv2.dilate(thresh, kernel, iterations=1)
	dilate_img = cv2.GaussianBlur(dilate_img, (3, 3), cv2.BORDER_DEFAULT)
	kernel = np.ones((1, 1), 'uint8')
	dilate_img = cv2.dilate(dilate_img, kernel, iterations=1)
	_, thresh = cv2.threshold(dilate_img, 252, 255, cv2.THRESH_BINARY)
	#showImage("Fattened", thresh,1,2000)
	return thresh


def showImage(winName, viewImage, viewDivFactor=8, ttime=400):
	my = int(viewImage.shape[0] / viewDivFactor)
	mx = int(viewImage.shape[1] / viewDivFactor)
	dsize = (mx, my)
	# resize image
	scaledImage = cv2.resize(viewImage, dsize, interpolation=cv2.INTER_AREA)
	cv2.imshow(winName, scaledImage)
	cv2.waitKey(2*ttime)
	cv2.destroyWindow(winName)
	return

def nearby(a, b, horMin, horMax, vertMin, vertMax):
	midptxa = (a[0] + a[2])/2
	midptya = (a[1] + a[3])/2
	#print(f'in nearby 1: {midptxa}\n')
	for pos in b:
		midptxb = (pos[0] + pos[2])/2
		midptyb = (pos[1] + pos[3])/2
		#print(f'in nearby 2: {midptxb}\n')
		if abs(midptxb - midptxa) < horMax:
			if abs(midptxb - midptxa) > horMin:
				if (midptyb - midptya) < vertMax:
					if (midptyb - midptya) > vertMin:
						print(f'abs(midptxa - midptxb) : {abs(midptxa - midptxb)}     (midptyb - midptya) < vertMax : {(midptyb - midptya)}   ')
						return pos
	return np.array([])


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
ap.add_argument("-g", "--threshold", required=True, help="0-1 for detection threshold cuttoff")
ap.add_argument("-c", "--color", required=True, help="red blue green magenta brown cyan yellow")
ap.add_argument("-s", "--shape", required=True, help="square circle")
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
ap.add_argument("-p", "--prepend",  required=True, help="prefix to add to input name to make output name")
ap.add_argument("-r", "--related",  required=True, help="which other template (t,m,l) this relates to")
ap.add_argument("-mm", "--mm",  required=True, help="xMax,xMin,yMin,yMax for mid relative to top")
ap.add_argument("-ll", "--ll",  required=True, help="xMax,xMin,yMin,yMax for low relative to top")
ap.add_argument("-d", "--delta",  required=True, help="tolerance in x and y position, pic pixels")
ap.add_argument("-z", "--pagemask",  required=True, help="filename to use for a pagemask to avoid title block")
ap.add_argument("-a", "--numtemplates",  required=True, help="number of templates")


args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges


if args['numtemplates'] == '3':
	templates = [args['template']+'_l.tif', args['template']+'_m.tif', args['template']+'_t.tif']
if args['numtemplates'] == '1':
	templates = [args['template'] + '.tif']

diffs = str(args['threshold']).split(',')

if args['color'] == 'red':
	color = (0, 0, 255)
if args['color'] == 'blue':
	color = (255, 0, 0)
if args['color'] == 'green':
	color = (0, 255, 0)
if args['color'] == 'magenta':
	color = (255, 0, 255)
if args['color'] == 'brown':
	color = (0, 255, 255)
if args['color'] == 'cyan':
	color = (255, 0, 200)
if args['color'] == 'yellow':
	color = (255, 255, 0)

print(args)
for imagePath in glob.glob(args["images"]):
	low = []
	mid = []
	top = []
	templatePos = []
	print(f"Image in : {imagePath}")
	image = cv2.imread(imagePath)
	if args['visualize']:
		showImage('main', image,8)
	clone = image.copy()
	orig = image.copy()
	checkprint = image.copy()
	raw = cv2.imread(imagePath)
	image = cv2.imread(imagePath, 0)
	edged = fatten(image.copy())
	if args['visualize']:
		showImage('fat11111', edged)
	edged = fatten(edged)
	if args['visualize']:
		showImage('fat22222', edged, 8)
	edged = fatten(edged)
	if args['visualize']:
		showImage('fat22222', edged, 8)
	edged = fatten(edged)
	if args['visualize']:
		showImage('fat22222', edged, 8)

	_, edged = cv2.threshold(edged, 254, 255, cv2.THRESH_BINARY)
	edged = cv2.bitwise_not((edged))
	if args['visualize']:
		showImage('image', edged,8)
	ccc = 0
	for templateName in templates:
		template = cv2.imread(templateName, 0)
		tH = template.shape[0]
		tW = template.shape[1]
		_, template = cv2.threshold(template, 254, 255, cv2.THRESH_BINARY)
		template = (fatten(template))
		template = (fatten(template))
		template = (fatten(template))
		template = (fatten(template))
		template = cv2.bitwise_not(template)
		#showImage("Template", template,1,3000)
		found = None
		# loop over the scales of the image
		for scale in np.linspace(1.0, 1.0, 1)[::-1]:
			print(f"Matching template {templateName}")
			#rawMask = cv2.imread(imagePath,0)
			#cv2.rectangle(rawMask, (0,0), (4965,3500), 0, -1)
			#cv2.rectangle(rawMask, (300, 300), (4000, 3200), 255, -1)
			#_, rawMask = cv2.threshold(rawMask, 254, 255, cv2.THRESH_BINARY)
			#mask = fatten(rawMask)
			#mask = cv2.bitwise_not(rawMask)
			#################################edged = cv2.bitwise_and(edged, mask)
			#edged = cv2.bitwise_and(edged)
			if args['visualize']:
				showImage('image', edged, 8, 1000)
			result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
			if args['visualize']:
				showImage("Result", result, 8)

			# find all locations in the result map where the matched value is
			# greater than the threshold, then clone our original image so we
			# can draw on it
			(yCoords, xCoords) = np.where(result >= float(diffs[ccc]))
			print(f"Threshold {float(diffs[ccc])}")
			print(f"[INFO] {len(yCoords)} matched locations *before* NMS")
			# loop over our starting (x, y)-coordinates
			for (x, y) in zip(xCoords, yCoords):
				# draw the bounding box on the image
				cv2.rectangle(clone, (x, y), (x + tW, y + tH),color, 3)
			# initialize our list of rectangles
			rects = []
			# loop over the starting (x, y)-coordinates again
			for (x, y) in zip(xCoords, yCoords):
				# update our list of rectangles
				rects.append((x, y, x + tW, y + tH))
			# apply non-maxima suppression to the rectangles
			pick = non_max_suppression(np.array(rects))
			print("[INFO] {} matched locations *after* NMS".format(len(pick)))
			# loop over the final bounding boxes
			if ccc == 0:  # lower
				low = pick.copy()
			if ccc == 1:
				mid = pick.copy()
			if ccc == 2:
				top = pick.copy()
			for (startX, startY, endX, endY) in pick:
				cv2.rectangle(raw, (startX, startY), (endX, endY), color, 15)
			if args['visualize']:
				showImage("After NMS", raw, 8)
			imagePath2 = args['prepend']+'_raw_' + str(ccc) + '_' + imagePath
			cv2.imwrite(imagePath2, raw)
			print(f'Check template file : {imagePath2}')
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
			# check to see if the iteration should be visualized

		ccc = ccc + 1
	if args['numtemplates'] == '3':
		for pos in top:
			tol = args['mm'].split(',')
			midMatch = nearby(pos, mid, int(tol[1]), int(tol[0]), int(tol[2]), int(tol[3]))
			if midMatch.any():
				cv2.rectangle(checkprint, ((int(pos[0])), (int(pos[1]))),((int(pos[2])), (int(pos[3]))),(255, 255, 0), 5)
				cv2.rectangle(checkprint, (int((pos[0]+pos[2])/2)-2, int((pos[1]+pos[3])/2)-2), (int((pos[0]+pos[2])/2)+2, int((pos[1]+pos[3])/2)+2), (255, 0, 255), 5)

				cv2.rectangle(checkprint, (midMatch[0], midMatch[1]), (midMatch[2], midMatch[3]), (0, 255, 255), 5)
				cv2.rectangle(checkprint, (int((midMatch[0]+midMatch[2])/2)-2, int((midMatch[1]+midMatch[3])/2)-2), (int((midMatch[0]+midMatch[2])/2)+2, int((midMatch[1]+midMatch[3])/2)+2), (0, 0, 255), 5)
				cv2.rectangle(checkprint, (int((pos[0] + pos[2]) / 2) - int(tol[1]), int((pos[1] + pos[3]) / 2) + int(tol[2])),(int((pos[0] + pos[2]) / 2) - int(tol[0]), int((pos[1] + pos[3]) / 2) + int(tol[3])),(128, 128, 128), 5)

			#cv2.rectangle(checkprint, ((int(pos[0]) + int(tol[1])), (int(pos[1]) + int(tol[3]))), ((int(pos[0]) + int(tol[0])), (int(pos[1]) + int(tol[2]))), (0, 0, 255), 5)
				#cv2.rectangle(checkprint, (int(mid[0][1]), int(mid[0][3])), (int(mid[0][0]) , int(mid[0][1])), (255, 0, 255), 5)
			tol = args['ll'].split(',')
			lowMatch = nearby(pos, low, int(tol[1]), int(tol[0]), int(tol[2]), int(tol[3]))
			if lowMatch.any():
				#cv2.rectangle(checkprint, ((int(pos[0])+int(tol[1])), (int(pos[1])+int(tol[3]))),((int(pos[2])+int(tol[0])), (int(pos[3])+int(tol[2]))),(0, 255, 0), 5)
				cv2.rectangle(checkprint, ((int(pos[0])), (int(pos[1]))),((int(pos[2])), (int(pos[3]))),(255, 255, 0), 5)
				cv2.rectangle(checkprint, (int((pos[0]+pos[2])/2)-2, int((pos[1]+pos[3])/2)-2), (int((pos[0]+pos[2])/2)+2, int((pos[1]+pos[3])/2)+2), (0, 0, 255), 5)

				cv2.rectangle(checkprint, (lowMatch[0], lowMatch[1]), (lowMatch[2], lowMatch[3]), (0, 255, 255), 5)
				cv2.rectangle(checkprint, (int((lowMatch[0]+lowMatch[2])/2)-2, int((lowMatch[1]+lowMatch[3])/2)-2), (int((lowMatch[0]+lowMatch[2])/2)+2, int((lowMatch[1]+lowMatch[3])/2)+2), (0, 0, 255), 5)
				cv2.rectangle(checkprint, (int((pos[0] + pos[2]) / 2) + int(tol[1]), int((pos[1] + pos[3]) / 2) + int(tol[2])),(int((pos[0] + pos[2]) / 2) + int(tol[0]), int((pos[1] + pos[3]) / 2) + int(tol[3])), (128, 128, 128), 5)

			#cv2.rectangle(checkprint, ((int(pos[1]) + int(tol[1])), (int(pos[3]) + int(tol[0]))),((int(pos[0]) + int(tol[2])), (int(pos[2]) + int(tol[3]))),(255, 255, 0), 5)
			if midMatch.any() and lowMatch.any():
				templatePos.append((pos[0], pos[1], lowMatch[2], lowMatch[3]))

	if args['numtemplates'] == '1':
		templatePos = low

	for t in templatePos:
		if args['shape'] == 'square':
			cv2.rectangle(checkprint, (t[0]-3, t[1]-3), (t[2]+3, t[3]+3), color, 16)
		if args['shape'] == 'circle':
			x= int((t[0] + t[2])/2)
			y= int((t[1] + t[3])/2)
			radius = abs(int((t[0] - t[2])/2))+5
			cv2.circle(checkprint,(x,y), radius, color, 16)

	showImage("Visualize", checkprint, 8 , 5000)
	print(f"Output file : {args['prepend']+imagePath}")
	cv2.imwrite(args['prepend']+imagePath, checkprint)
	idx = 0
	for t in templatePos:
		extract = orig[t[1]+10:t[3]-10,t[0]+10:t[2]-10]
		#extract = cv2.bitwise_not(extract)
		extractbw = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)
		im = extract

		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

		digits = []
		for cnt in contours:
			idx += 1
			x, y, w, h = cv2.boundingRect(cnt)
			#roi = im[y:y + h, x:x + w]
			#cv2.imwrite(str(idx) + '.jpg', roi)
			if h>43 and h<48:
				roi = im[y:y + h, x:x + w]
				cv2.imwrite(str(idx) + '.png', roi)
				digits.append((y*1000+x, x,y,w,h,im[y:y + h, x:x + w]))

		sortedDigits = sorted(digits)
		for d in sortedDigits:
			#showImage('digit', im[y:y + h, x:x + w],1,100)
			##cv2.rectangle(d[5], (,  d[2]), (d[1] + d[3], d[2] + d[4]), (200, 0, 0), 2)
			win = 0
			winner = -1
			for fd in range(0,9):
				img_cmp = cv2.imread(f'{fd}.png')
				blank = np.ones((h, h, 3), dtype = np.uint8)
				blank = 255 * blank
				height = img_cmp.shape[0]
				width = img_cmp.shape[1]
				blank[0:height,0:width] = img_cmp
				#showImage('Blank', blank, 1, 5000)
				a = fatten(cv2.bitwise_not(blank))
				b = fatten(cv2.bitwise_not(d[5]))
				result = cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)
				(yCoords, xCoords) = np.where(result >= float(0.45))
				if len(yCoords)>win:
					winner = fd

			print(f'{winner}', end='')

		print()
		if args['visualize']:
			showImage('extract', im, 1, 100)
		digits.sort()
		#print(digits)

	print("\n\n")


