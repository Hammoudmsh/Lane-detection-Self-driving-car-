import cv2
import numpy as  np
import matplotlib.pyplot as plt
import imageLaneDetector as ild


if __name__=="__main__":

	cap = cv2.VideoCapture('test2.mp4')

	while (cap.isOpened()):
		_, frame = cap.read()
		result  = ild.laneDetector(frame, (255,255,255))
		cv2.imshow("Road imag", result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
