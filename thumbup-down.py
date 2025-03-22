#CURRENTLY WORKING FOR THUMBS UP AND THUMBS DOWN

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Define a function to classify gestures based on hand landmarks
def recognize_gesture(landmarks):
    # Thumbs Up: Thumb tip higher than thumb base
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    if thumb_tip.y < thumb_base.y:
        return "Thumbs Up"

    # Thumbs Down: Thumb tip lower than thumb base
    if thumb_tip.y > thumb_base.y:
        return "Thumbs Down"

    return "Unknown Gesture"


# Start video capture
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better mirror image
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB as MediaPipe requires RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize the gesture based on landmarks
            gesture = recognize_gesture(landmarks.landmark)

            # Display the gesture on the frame
            cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the recognized gesture
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
