import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe 손 추적 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 빈 캔버스 생성
canvas = np.ones((720, 1280, 3), dtype='uint8') * 255  # 흰 배경
drawing = False  # 그리기 상태 변수

# 엄지와 검지 사이의 거리 계산 함수
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# 웹캠 비디오 캡처
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe로 손 랜드마크 추출
    results = hands.process(image)

    # 이미지를 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손의 랜드마크를 이미지 위에 그리기 (디버그 용도)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 엄지(4번)와 검지(8번) 랜드마크 좌표 추출
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # 이미지 좌표로 변환
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # 엄지와 검지 사이의 거리 계산
            distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

            # 엄지와 검지가 가까워지면 그리기 시작 (거리 기준은 약 30 픽셀 이하)
            if distance < 30:
                drawing = True
            else:
                drawing = False

            # 그리기 상태일 때 원을 그려 자유롭게 그림 그리기
            if drawing:
                cv2.circle(canvas, (index_x, index_y), 5, (0, 0, 0), -1)  # 검은색 작은 원

    # 결과 화면에 출력 (실시간 웹캠 영상과 캔버스)
    combined_image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
    cv2.imshow('Hand Drawing', combined_image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()