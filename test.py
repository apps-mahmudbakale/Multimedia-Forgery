import cv2

TrDict = {'csrt': cv2.TrackerCSRT_create,
		 'kcf': cv2.TrackerKCF_create,
		 'boosting': cv2.TrackerBoosting_create,
		 'mil': cv2.TrackerMIL_create,
		 'tld': cv2.TrackerTLD_create,
		 'medianflow': cv2.TrackerMedianFlow_create,
		 'mosse': cv2.TrackerMOSSE_create}	

tracker = TrDict['csrt']()

v = cv2.VideoCapture(r'movie.mp4')

ret, frame = v.read()

cv2.imshow('Frame', frame)

bb = cv2.selectROI('Frame', frame)

tracker.init(frame,bb) 