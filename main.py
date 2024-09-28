import cv2
import mediapipe as mp
import math
import numpy as np

def distance(pt1, pt2):
    return math.dist((pt1.x,pt1.y),(pt2.x,pt2.y))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

width, height = 640, 480
board = [[ 0 for _ in range(height)]for _ in range(width)]
lst = [0 for _ in range(width*height+1)]
n=1

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

hands = mp_hands.Hands()
while True:
    ret, frame = cap.read()
    if not ret:
        print("cammera error")
        break

    frame = cv2.flip(frame,1)
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                frame,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS
            )
        point = hand_landmark.landmark
        action = distance(point[4],point[8])<0.07
        action_point = (math.floor((point[4].x+point[8].x)/2*width),math.floor((point[4].y+point[8].y)/2*height))
                
        if action:
             cv2.circle(frame,action_point,5,(0,0,255),-1)
             board[action_point[0]][action_point[1]] = n
             lst[n]=action_point
             n=n+1
        for i in range(1,n):
             start = lst[i]
             end = lst[i+1] if lst[i+1] != 0 else lst[i]
             print(start,end)
             cv2.line(frame,start,end,(0,0,255),5)
             
    
    
    
    
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠 리소스 해제
cap.release()
cv2.destroyAllWindows()