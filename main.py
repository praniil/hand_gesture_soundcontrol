import cv2
import numpy as np 

#cascade classifier for detecting hands
hand_cascade = cv2.CascadeClassifier('home/pranil/.local/lib/python3.10/site-packages/cv2/data/haarcascade_hand.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, 1.1, 4)

    #draw rectanges around the detected hands
    for(x, y, w, z) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #display the frame
    cv2.imshow('Hand detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()