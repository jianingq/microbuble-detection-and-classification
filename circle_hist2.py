import argparse
import cv2
import numpy as np

def fill_holes(imInput):
	"""
	The method used in this function is found from
	https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

	"""

	# Threshold.
	thImg = cv2.adaptiveThreshold(imInput,255,cv2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	cv2.imshow("thImg", thImg)

	kernel = np.ones((2,2),np.uint8)

	erode = cv2.erode(thImg, kernel, iterations=1)
	cv2.imshow("erode", erode)

	# Copy the thresholded image.
	imFloodfill = erode#thImg.copy()

	# Get the mask.
	h, w = thImg.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

   
	# Floodfill from point (0, 0).
	cv2.floodFill(imFloodfill, mask, (0,0), (0,0,0))
	#cv2.floodFill(imFloodfill, mask, (230,617), (0,0,0))
	#cv2.imshow("imFloodfill", imFloodfill)

	return imFloodfill,thImg

if __name__ == "__main__":
	from os import listdir
	from os.path import isfile, join
	##onlyfiles = [f for f in listdir("./covid microbubbles") if f.endswith("jpg")]
	onlyfiles = [f for f in listdir("./ai_cropped/") if f.endswith("_cropped.png")]

	print(onlyfiles)
	for f in ["IMG_1583_cropped.png"]:#onlyfiles:#["IMG_1595_cropped.png","IMG_1540_cropped.png","IMG_3083_cropped.png","IMG_3021_cropped.png"]:#onlyfiles:#["18-strong+.jpg","11-medium+.jpg","5-medium+.jpg"]:#onlyfiles:
		print(f)
		# Load the image.
		image = cv2.imread("./ai_cropped/"+f)
		image = cv2.resize(image, (int(640), int(640)))#(int(480), int(480)))
		image2 = image.copy()
		#cv2.imshow("Original image", image)

		# Convert the image into grayscale image.
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		#cv2.imshow("Blurred", blurred)

		# Fille the "holes" on the image.
		filled,thImg = fill_holes(blurred)
		cv2.imshow("Filled", filled)


		kernel = np.ones((3, 3), np.uint8)
		#filled = cv2.erode(filled, np.ones((2, 2), np.uint8), iterations=1)
		#cv2.imshow("erode", filled)
		filled = cv2.dilate(filled, kernel, iterations=2)
		cv2.imshow("dilate", filled)

		edges = cv2.Canny(filled,100,200, apertureSize=3)#
		cv2.imshow("edges", edges)

		contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		contours = sorted(contours, key= lambda x:cv2.boundingRect(x)[0])#contours.sort(key=lambda x:cv2.boundingRect(x)[0])
		
		white_image = image.copy()
		white_image.fill(255)


		cv2.drawContours(white_image, contours,  -1, (255,0,0), 2)
		white_image = cv2.cvtColor(white_image,cv2.COLOR_BGR2GRAY)
		cv2.imshow('Objects Detected',white_image)
		"""

		contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		
		contour_list = []
		for contour in contours:
		    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		    area = cv2.contourArea(contour)
		    if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
		        contour_list.append(contour)

		cv2.drawContours(white_image, contour_list,  -1, (255,0,0), 2)
		white_image = cv2.cvtColor(white_image,cv2.COLOR_BGR2GRAY)
		cv2.imshow('Objects Detected',white_image)
		"""
		
		a = []
		b = []
		ii = 1

		for c in contours:
			(x,y),r = cv2.minEnclosingCircle(c)
			center = (int(x),int(y))
			r = int(r)
			if r >= 3 and r < 30:#r<=30:
				cv2.circle(image,center,r,(0,255,0),2)
				a.append(r)
				b.append(center)

		b.append([313, 313])
		b.append([121, 505])
		b.append([70, 262])

		a.append(121)
		a.append(121)
		a.append(121)

		image = cv2.resize(image, (int(640), int(640)))
		cv2.imshow("preprocessed", image)
		"""
		
		circles = cv2.HoughCircles(thImg, cv2.HOUGH_GRADIENT, 0.01, 10, param1 = 15, param2 = 15, minRadius = 0, maxRadius = 20)

		# Draw circles on the original image.
	 	
		if circles is not None:
			for i in range(circles.shape[1]-200):
				c = circles[0, i]

				cv2.circle( image, (c[0], c[1]), c[2], (0, 255, 0), 2)
				print("i = %d, r = %f" % (i, c[2]))

			cv2.imshow("Marked", image)
		else:
			print("circle is None")
		"""
		#cv2.imwrite(f.split(".")[0]+".jpg",image)

		# Block the execution.
		##cv2.waitKey(0)
		
		
		cv2.destroyAllWindows()
		print(a)
		np.save("./ai_cropped/neg/"+f.split(".")[0]+"_center.npy",b)
		
		np.save("./ai_cropped/neg/"+f.split(".")[0]+".npy",a)
