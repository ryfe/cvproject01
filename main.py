import cv2
from cvzone.HandTrackingModule import HandDetector as HD

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

detector = HD(maxHands=2,detectionCon=0.8)

while cv2.waitKey(33) < 0:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    hands,img = detector.findHands(frame)

    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        # Count the number of fingers up for the first hand
        #fingers1 = detector.fingersUp(hand1)
        #print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

        # Calculate distance between specific landmarks on the first hand and draw it on the image
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img)
        print(lmList1[8][:])

        # Check if a second hand is detected
        if len(hands) == 2:
            # Information for the second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]

            # Count the number of fingers up for the second hand
            fingers2 = detector.fingersUp(hand2)
            #print(f'H2 = {fingers2.count(1)}', end=" ")

            # Calculate distance between the index fingers of both hands and draw it on the image
    cv2.imshow("video",frame)

cap.release()
cv2.destroyAllWindows()