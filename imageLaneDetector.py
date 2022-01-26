import cv2
import numpy as  np
import matplotlib.pyplot as plt



def canny(img):
	# convert to gray
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# reduce noise using Gaussian filter
	blur = cv2.GaussianBlur(gray, (5,5),0)#change (5,5)
	# edges detction
	edges = cv2.Canny(blur,50, 150)#Canny internally do Gaussian filter
	return edges

def ROI(img):
	height = img.shape[0]
	width = img.shape[1]
	polygons = np.array([
		[(200,height),(1100,height),(550,250)]
		])
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, polygons,255)# fill area bu several polygons, s [] 
	masked_img = cv2.bitwise_and(img, mask)
	return masked_img


def display_lines(img, lines, color):
	line_img = np.zeros_like(img)
	if lines is not None:
		for line in lines:
			#print(line) line is 2D: [[x1, y1, x2, y2]]
			x1, y1, x2, y2 = line #line.reshape(4)
			cv2.line(line_img, (x1,y1), (x2,y2),color, 10)
		#cv2.line(line_img, (xl1,yl1), (xr1+100,yr1+100),(0,0,255), 10)

	return line_img

def make_coordinates(img, line_parameters):
	slope, intercept = line_parameters
	y1 = img.shape[0]
	y2 = int(y1 * (3/5))
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(lane_img, lines):
	left_fit = []
	right_fit = []

	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1)#get m , p
		slope, intercept = parameters
		if slope < 0:
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))
	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)
	left_line = make_coordinates(lane_img, left_fit_average)
	right_line = make_coordinates(lane_img, right_fit_average)
	return np.array([left_line, right_line])


def laneDetector(image, color):
	lane_img = image.copy()
	canny_img = canny(lane_img)
	# edges detction
	#plt.imshow(canny)
	ROI_img  = ROI(canny_img)
	lines = cv2.HoughLinesP(ROI_img, 2, np.pi / 180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)#<40 reject, <100 no
	averaged_lines = average_slope_intercept(lane_img, lines)
	line_img = display_lines(lane_img, averaged_lines, color)
	combo_img = cv2.addWeighted(line_img, 0.8, lane_img, 1, 1)
	
	xl1, yl1, xl2, yl2 = averaged_lines[0]
	xr1, yr1, xr2, yr2 = averaged_lines[1]
	
	xl = (xl1 + xl2)//2
	yl = (yl1 + yl2)//2
	xr = (xr1 + xr2)//2
	yr = (yr1 + yr2)//2
	
	xc, yc = (xl + xr)//2, (yl + yr)//2 
	
	cv2.line(combo_img, (xl, yl), (xr, yr),color, 2)
	cv2.line(combo_img, (xc, yc-20), (xc, yc+20),color, 2)

	return combo_img
"""
y= mx + p     y = f(x)
Hough space  b f(m) instead of lin,  we have dot
single point in x,y have inf lines = line in hough space
"""

if __name__=="__main__":
	image = cv2.imread('1.jpg')
	result = laneDetector(image, (255,255,255))
	cv2.imshow("Road imag", result)
	#plt.show()
	cv2.waitKey(0)
