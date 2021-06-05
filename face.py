from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import csv
import cv2
import numpy as np
import dlib
import math
from time import sleep
import train_model






vs = VideoStream(src=0).start()        
#  INISIALISAI FILE PRE-TRAINNED UNTUK DETEKSI WAJAH DAN DETEKSI LANDMARKS
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    frame = vs.read()
    #frame = imutils.rotate(frame,180 , scale = 1)
    frame = imutils.resize(frame, width = 480, height = 368)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 5)

    
    for i,face in enumerate(faces):
        
        for (x, y, w, h) in faces:
        
            face = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            
            
            # PENDETEKSIAN LANDMARKS WAJAH 

            landmarks = predictor(gray,face)
            x_eye=[]
            y_eye=[]
            x_nose=[]
            y_nose=[]
        
            # PEMBERIAN TITIK DAN GARIS PADA LANDMARKS WAJAH

            cv2.rectangle(frame, (x,y), (x+w,y+h),(0, 255, 0),2)      
            left_pointA = (landmarks.part(36).x, landmarks.part(36).y)
            right_pointB = (landmarks.part(45).x, landmarks.part(45).y)
            up_pointA = (landmarks.part(27).x, landmarks.part(27).y)
            be_pointB = (landmarks.part(30).x, landmarks.part(30).y)
            line = cv2.line(frame, left_pointA, right_pointB, (255, 0,0),1)
            line2 = cv2.line(frame, up_pointA, be_pointB, (255, 0,0),1)
            line2 = cv2.line(frame, left_pointA, be_pointB, (255, 0,0),1)
            line2 = cv2.line(frame, right_pointB, be_pointB, (255, 0,0),1)
            cv2.circle(frame, (landmarks.part(36).x, landmarks.part(36).y), 1, (0,0,255), thickness=2)
            cv2.circle(frame, (landmarks.part(45).x, landmarks.part(45).y), 1, (0,0,255), thickness=2)
            cv2.circle(frame, (landmarks.part(27).x, landmarks.part(27).y), 1, (0,0,255), thickness=2)
            cv2.circle(frame, (landmarks.part(30).x, landmarks.part(30).y), 1, (0,0,255), thickness=2)

        	# EKSTRAKSI NILAI FITUR YANG DIGUNAKAN

            x_eye.append(float(landmarks.part(36).x))
            x_eye.append(float(landmarks.part(45).x))
            y_eye.append(float(landmarks.part(36).y))
            y_eye.append(float(landmarks.part(45).y))
            x_nose.append(float(landmarks.part(27).x))
            x_nose.append(float(landmarks.part(30).x))
            y_nose.append(float(landmarks.part(27).y))
            y_nose.append(float(landmarks.part(30).y))


            # PENGHITUNGAN NILAI DENGAN RUMUS EUCLIDEAN DISTANCE
            eye_dist = math.sqrt((x_eye[0] - x_eye[1])**2 - (y_eye[0] - y_eye[1])**2)
            nose_dist = math.sqrt((y_nose[0] - y_nose[1])**2 - (x_nose[0] - x_nose[1])**2)
            area = 0.5 * eye_dist * nose_dist

            data_test =[]
        
          	
            data_test.append(w)
            data_test.append(eye_dist)
            data_test.append(nose_dist)
            data_test.append(area)
            
            # PENGENALAN WAJAH YANG DI DETEKSI
            print("w: {} Eye Distance: {} Nose Distance: {} Area: {}".format(
            w , eye_dist , nose_dist , area))
            print("=====================================================================================")
            data_test = np.asarray(data_test)
            data_test = data_test.reshape(1,-1)
            result = train_model.predict(data_test)
            
            
            print ("Hasil Prediksi Dengan SVM : ", result)
            cv2.putText(frame, "Face Widht : ", (0,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame, "Eye Distance : ", (0,40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame, "Nose Distance : ", (0,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame, "Area : ", (0,80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame, str(w), (100,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame, str(eye_dist), (115,40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame,str(nose_dist), (125,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.putText(frame, str(area), (50,80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            if result == 1:
                cv2.putText(frame, "USER ", (x-10,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
            else:
                cv2.putText(frame, "UNKNOWN ", (x-10,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

           
            
                

    cv2.imshow("HASIL", frame)
    key = cv2.waitKey(1)
    
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()




