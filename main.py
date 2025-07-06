import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset_path = 'hand_sign_landmarks.csv'
dataset = pd.read_csv(dataset_path)

# Prepare the data
X = dataset.iloc[:, 1:].values  # Features: all columns except the first
y = dataset.iloc[:, 0].values   # Labels: the first column

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
accuracy = np.mean(cv_scores)
print(f"Cross-validation accuracy: {accuracy:.2f}")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to recognize gesture based on landmarks and calculate confidence
def recognize_gesture_and_confidence(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)
    landmarks_scaled = scaler.transform(landmarks)
    proba = rf.predict_proba(landmarks_scaled)[0]
    gesture = rf.predict(landmarks_scaled)[0]
    confidence = np.max(proba)  # Confidence of the predicted gesture
    return str(gesture), confidence

# Initialize webcam
cap = cv2.VideoCapture(1)

# Prepare the text to display accuracy
accuracy_text = f"Model Accuracy: {accuracy:.2f}"

# Define font properties
font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 2

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from webcam")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to a list of tuples
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Recognize gesture and get confidence score
            gesture, confidence = recognize_gesture_and_confidence(landmarks)

            print(f"Recognized gesture: {gesture}, Confidence: {confidence:.2f}")  # Debugging statement

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box coordinates
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Draw the bounding box
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

            # Display the gesture text and confidence score inside the bounding box with bold effect
            text = f"{gesture} ({confidence:.2f})"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x_min
            text_y = y_min - 10

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i != 0 or j != 0:
                        cv2.putText(frame, text, (text_x + i, text_y + j), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    else:
        print("No hand landmarks detected")  # Debugging statement

    # Show the frame with accuracy text
    cv2.putText(frame, accuracy_text, (10, 30), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Check if Enter key (ASCII code 13) is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()