import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

label = input("Masukkan nama gesture: ")
path = f"dataset/SIBI/{label}"

os.makedirs(path, exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand.landmark:
                data.append(lm.x)
                data.append(lm.y)

            np.save(f"{path}/{count}.npy", np.array(data))
            count += 1

            print("Data ke:", count)

    cv2.imshow("Ambil Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()