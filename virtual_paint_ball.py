import cv2
import mediapipe as mp
import numpy as np

# Set up the game window dimensions
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 10

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start the webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas (background)
canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255  # White background

# Define the paintball color (red)
paintball_color = (0, 0, 255)  # Red color (BGR format)

# Function to get hand position
def get_hand_position(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the wrist landmark position (the base of the hand)
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH)
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT)
            return wrist_x, wrist_y
    return None, None

# Game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the canvas to match the frame size (in case they are different)
    frame_height, frame_width, _ = frame.shape
    canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # White background

    # Get hand position
    hand_x, hand_y = get_hand_position(frame)
    
    if hand_x is not None and hand_y is not None:
        # Draw the paintball (where the hand is located)
        cv2.circle(canvas, (hand_x, hand_y), BALL_RADIUS, paintball_color, -1)  # Paint the ball
        
        # Draw the wrist (base of the hand) for visual feedback
        cv2.circle(frame, (hand_x, hand_y), 15, (0, 255, 0), -1)  # Green color to show where the hand is

    # Combine the canvas and the webcam frame to show the effect
    output_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Display the canvas with paintballs
    cv2.imshow("Virtual Paint Ball", output_frame)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

# Release the webcam and close the game window
cap.release()
cv2.destroyAllWindows()
