import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from PIL import ImageGrab

# Initialize MediaPipe hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Variables for tracking time of gestures
last_click_time = time.time()
click_timeout = 0.5

# Screen size
screen_width, screen_height = pyautogui.size()

# Function to convert screen coordinates
def normalize_to_screen(x, y):
    x = int(np.interp(x, [0, 1], [0, screen_width]))
    y = int(np.interp(y, [0, 1], [0, screen_height]))
    return x, y

# Function to take a screenshot
def take_screenshot():
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    print("Screenshot saved!")

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for better user experience
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Get landmarks for the index finger and thumb
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Calculate positions of the index finger and thumb in screen coordinates
            index_x, index_y = normalize_to_screen(index_finger_tip.x, index_finger_tip.y)
            thumb_x, thumb_y = normalize_to_screen(thumb_tip.x, thumb_tip.y)
            
            # Move the mouse cursor based on the index finger
            pyautogui.moveTo(index_x, index_y)
            
            # Detect the gestures based on distance and angles
            distance = np.linalg.norm([index_x - thumb_x, index_y - thumb_y])
            
            # Perform left click if the distance between index and thumb is small
            if distance < 30:  # Adjust the threshold as needed
                pyautogui.click()
                if time.time() - last_click_time < click_timeout:
                    pyautogui.doubleClick()
                last_click_time = time.time()
            
            # Perform right click if the index finger and thumb form a "V"
            elif distance < 50:  # Adjust the threshold as needed
                pyautogui.rightClick()

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with gestures visualized
    cv2.imshow('Virtual Mouse', frame)

    # Check for "Q" key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check for gesture to take a screenshot
    if time.time() - last_click_time > 3:  # If the user holds the hand for more than 3 seconds
        take_screenshot()

cap.release()
cv2.destroyAllWindows()
