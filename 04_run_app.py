import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Load your trained model
try:
    model = pickle.load(open("gesture_recognizer.pkl", "rb"))
    print("Model loaded successfully ✅")
except FileNotFoundError:
    print("Error: Could not find 'gesture_recognizer.pkl'. Make sure you trained the model!")
    exit()

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    max_num_hands=1
)

cap = cv2.VideoCapture(0)

# UI Variables
last_prediction = "Waiting for sign..."
current_confidence = 0.0

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            row = []
            
            # --- THE 3D MATH UPGRADE ---
            # Grab the wrist as the anchor point (X, Y, AND Z)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            # Extract 3D coordinates relative to the wrist
            for lm in hand_landmarks.landmark:
                row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
            # Convert to numpy array for the model
            X_input = np.array([row])

            # --- THE CONFIDENCE UPGRADE ---
            # Get probabilities for ALL classes instead of forcing a guess
            probabilities = model.predict_proba(X_input)[0]
            max_prob_index = np.argmax(probabilities)
            
            confidence = probabilities[max_prob_index]
            prediction = model.classes_[max_prob_index]

            # Only update the screen if the AI is at least 80% confident
            # AND if the prediction isn't "IDLE"
            if confidence > 0.80 and prediction != "IDLE":
                if prediction != last_prediction:
                    last_prediction = prediction
                current_confidence = confidence
            elif prediction == "IDLE" and confidence > 0.80:
                 last_prediction = "..."
                 current_confidence = 0.0

    # --- THE UI DESIGN ---
    # Draw a black background box for readability
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
    
    # Show Translation
    cv2.putText(frame, f"Translation: {last_prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show Confidence %
    if current_confidence > 0:
        cv2.putText(frame, f"Confidence: {int(current_confidence * 100)}%", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.imshow("Voice2Sense AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()