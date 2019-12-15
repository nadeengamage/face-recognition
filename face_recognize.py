import cv2, sys, os
import numpy as np

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Recognizing Face Please Be in sufficient Lights...') 

(images, lables, names, id) = ([], [], {}, 0) 
for (subdirs, dirs, files) in os.walk(datasets): 
	for subdir in dirs: 
		names[id] = subdir 
		subjectpath = os.path.join(datasets, subdir) 
		for filename in os.listdir(subjectpath): 
			path = subjectpath + '/' + filename 
			lable = id
			images.append(cv2.imread(path, 0)) 
			lables.append(int(lable)) 
		id += 1
(width, height) = (130, 100) 

(images, lables) = [np.array(lis) for lis in [images, lables]] 

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables) 

face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 


while True: 
	(_, im) = webcam.read() 
	(_, im2) = webcam.read() 

	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])

	for (x, y, w, h) in faces: 
		cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		
		prediction = model.predict(face_resize) 
		cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3) 
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 

		if prediction[1]<100:
		    cv2.putText(gray, 'The person of % s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2) 
		    cv2.putText(im, 'The person of % s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2) 
		else: 
		    cv2.putText(gray, 'Not Recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
		    cv2.putText(im, 'Not Recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	cv2.imshow('Window 1', im)
	cv2.imshow('Window 2', im2)
	cv2.imshow('Window 3', gray)
	cv2.imshow('Window 4', mask)
	
	key = cv2.waitKey(10) 
	if key == 27: 
		cv2.destroyAllWindows()
		break
