import cv2
import mediapipe as mp
import math

def distance(pt1, pt2):
    math.dist((pt1.x,pt1.y),(pt2.x,pt2.y))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

hands = mp_hands.Hands()

while True:
    ret, frame = cap.read()
    if not ret:
        print("cammera error")
        break

    frame = cv2.flip(frame,1)
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hands_landmarks:
        for hand_landmark in result.multi_hands_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS
            )

        point = hand_landmark.landmark
