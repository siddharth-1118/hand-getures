import cv2
import mediapipe as mp
import csv
import os
import time

CSV_FILE = "gesture_dataset.csv"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Create the CSV and write headers if it doesn't exist
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        header.append('label')
        writer.writerow(header)

print("\n=== VOICE2SENSE SMART DATA COLLECTOR ===")
print("What gesture are you recording? (e.g., A, B, HELP, CALL, or IDLE)")
GESTURE_LABEL = input("Enter label: ").strip().upper()

cap = cv2.VideoCapture(0)

# Variables for the timer
is_recording = False
start_time = 0
RECORDING_DURATION = 10  # Seconds

print(f"\n[READY] Preparing to record data for: {GESTURE_LABEL}")
print("Press 'R' ONCE to start the 10-second timer. Press 'Q' to quit early.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Start recording if 'R' is pressed and we aren't already recording
    if key == ord('r') and not is_recording:
        is_recording = True
        start_time = time.time()
        print(f"Recording started! Do the '{GESTURE_LABEL}' gesture...")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # If the timer is running, extract and save the data
            if is_recording:
                elapsed_time = time.time() - start_time
                time_left = RECORDING_DURATION - elapsed_time

                if time_left > 0:
                    row = []
                    # WRIST NORMALIZATION: Get the wrist coordinates (X, Y, Z)
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z
                    
                    # Subtract wrist position from every joint to get pure shape
                    for landmark in hand_landmarks.landmark:
                        row.extend([landmark.x - base_x, landmark.y - base_y, landmark.z - base_z])
                    row.append(GESTURE_LABEL)

                    # Save to CSV
                    with open(CSV_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    # Draw visual countdown timer
                    cv2.putText(frame, f"RECORDING: {int(time_left) + 1}s", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    # Time is up!
                    print(f"Successfully recorded 10 seconds of data for '{GESTURE_LABEL}'.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit() # Automatically exit the script so you can run it again easily

    # Standard UI text
    if not is_recording:
        cv2.putText(frame, f"Target: {GESTURE_LABEL} | Press 'R' to Start", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Voice2Sense - Data Collector', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()