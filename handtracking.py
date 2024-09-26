import cv2
import mediapipe as mp

# Mediapipe 모듈 초기화
mp_hands = mp.solutions.hands  # 손 추적 모듈
mp_drawing = mp.solutions.drawing_utils  # 랜드마크를 그리기 위한 유틸리티

# 웹캠을 통한 비디오 스트림 캡처
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# 손 추적 설정
with mp_hands.Hands(
        max_num_hands=2,  # 추적할 손의 최대 수
        min_detection_confidence=0.7,  # 감지 신뢰도
        min_tracking_confidence=0.7  # 추적 신뢰도
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            break

        # BGR 이미지를 RGB로 변환 (Mediapipe는 RGB 이미지를 사용)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능을 위해 이미지를 쓰기 방지 모드로 설정
        image.flags.writeable = False

        # 손을 감지
        results = hands.process(image)

        # 원래 이미지로 되돌리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 손이 감지되었을 경우
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크를 이미지에 그리기
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        image = cv2.flip(image,1)
        # 결과 이미지 출력
        cv2.imshow('Hand Tracking', image)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠 리소스 해제
cap.release()
cv2.destroyAllWindows()