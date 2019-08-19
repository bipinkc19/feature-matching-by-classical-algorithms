import numpy as np
import cv2

cap = cv2.VideoCapture('./videos/VID_20190819_125759.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # lower_blue = np.array([50,160,50])
    # upper_blue = np.array([130,255,255])
    
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,127,255,0)
    gray2 = gray.copy()
    mask = np.zeros(gray.shape,np.uint8)
    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv2.contourArea(cnt)<5000:
            cv2.drawContours(frame,[cnt],0,(0,255,0),2)
            cv2.drawContours(mask,[cnt],0,255,-1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
