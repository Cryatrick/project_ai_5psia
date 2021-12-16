import cv2
import numpy as np
import os 
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "cascade/haarcascade_frontalface_default.xml"
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
file1 = open('name_data.txt', 'r')
Lines = file1.readlines()
 
count = 0
names = [None] * len(Lines)
for line in Lines:
    count += 1
    curr_string = line.strip()
    curr_data = curr_string.split(';')
    names[int(curr_data[0])] = str(curr_data[1])
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 1028) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    emo_faces = faces
    if len(emo_faces) > 0:
        emo_faces = sorted(emo_faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = emo_faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        emotion_label = EMOTIONS[preds.argmax()]
    else: continue
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        name_string = "Name : "+str(id)
        emo_string = "Mood : "+str(emotion_label)
        cv2.putText(img, name_string, (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, emo_string, (x+5,y+h-5), font, 1, (255,255,0), 1)  
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        # draw the label + probability bar on the canvas
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
        (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 2)
    cv2.imshow('Camera',img) 
    cv2.imshow("Mood Probabilities", canvas)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()