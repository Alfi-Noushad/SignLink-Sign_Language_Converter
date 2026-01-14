import os
import cv2
import mediapipe as mp
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR = '../data'
LABEL = "B"  # Change this for each of your 50 signs
DATA_SIZE = 500     # Number of frames to capture per sign

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create a subfolder for the specific sign
class_dir = os.path.join(DATA_DIR, LABEL)
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
counter = 0

print(f"Ready to collect data for {LABEL}. Press 's' to start.")

while counter < DATA_SIZE:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    cv2.putText(frame, f'Collecting: {LABEL} | Count: {counter}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Data Collection', frame)

    # Press 's' to start recording landmarks
    if cv2.waitKey(25) & 0xFF == ord('s') or counter > 0:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                # Normalize relative to wrist (landmark 0)
                x0 = hand_landmarks.landmark[0].x
                y0 = hand_landmarks.landmark[0].y
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - x0)
                    data_aux.append(y - y0)
                
                # Save as a single row in a text file inside the folder
                with open(os.path.join(class_dir, f'{counter}.txt'), 'w') as f:
                    f.write(",".join(map(str, data_aux)))
                
                counter += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()