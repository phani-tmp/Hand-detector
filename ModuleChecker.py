import cv2 
import mediapipe as mp 
import HandDetectorModule
Detector=HandDetectorModule.handDetector()
cap=cv2.VideoCapture(0)
while True: 
    ret,Img=cap.read() 
    img=Detector.findHands(Img )
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()