import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the saved model
model_dict = pickle.load(open('../model/model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x0, y0 = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - x0)
                data_aux.append(lm.y - y0)

            # 1. Get the probability (confidence) score
            prediction_probs = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(prediction_probs) * 100  # Highest probability
            
            # 2. Get the predicted label
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            # 3. Create the text string (e.g., "HELLO: 95%")
            display_text = f"{predicted_character}: {confidence:.1f}%"

            # 4. Display in BLACK color (0, 0, 0)
            cv2.putText(frame, display_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Live Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()