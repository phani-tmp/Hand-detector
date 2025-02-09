# import cv2
# import mediapipe as mp
# import time
# class handDetector():
#     def __init__(self, mode=False, maxHands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.detectionCon = float(min_detection_confidence)
#         self.trackCon = float(min_tracking_confidence)
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.maxHands,
#                                         self.min_detection_confidence, self.min_tracking_confidence)
#         self.mpDraw = mp.solutions.drawing_utils
#     def findHands(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)
#         # print(results.multi_hand_landmarks)
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(img, handLms,
#                                                self.mpHands.HAND_CONNECTIONS)
#         return img
#     def findPosition(self, img, handNo=0, draw=True):
#         lmList = []
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]
#             for id, lm in enumerate(myHand.landmark):
#                 # print(id, lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # print(id, cx, cy)
#                 lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
#         return lmList
# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = handDetector()
#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
#         if len(lmList) != 0:
#             print(lmList[4])
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 255), 3)
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
# if __name__ == "__main__":
#     main()
import cv2
import mediapipe as mp
import time

class handDetector:
    def __init__(self, mode=False, maxHands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(min_detection_confidence)
        self.trackCon = float(min_tracking_confidence)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None  # Initialize results

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):  # Prevent index error
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            continue  # Skip the frame if there's an error

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if lmList:
            print(lmList[4])  # Print thumb tip position

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
