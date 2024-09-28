import cv2
import mediapipe as mp

# Mediapipe 모듈 초기화
mp_hands = mp.solutions.hands  # 손 추적 모듈
mp_drawing = mp.solutions.drawing_utils  # 랜드마크를 그리기 위한 유틸리티

# 웹캠을 통한 비디오 스트림 캡처
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# 손 추적 설정
hands =  mp_hands.Hands( )

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break
    frame = cv2.flip(frame,1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 손이 감지되었을 경우
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크를 이미지에 그리기
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # 결과 이미지 출력
    cv2.imshow('Hand Tracking', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 리소스 해제
cap.release()
cv2.destroyAllWindows()