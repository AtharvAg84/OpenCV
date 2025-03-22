import cv2
import mediapipe as mp
import numpy as np
import random

# Set up the game window dimensions
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 20
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 20

# Game variables
ball_x = random.randint(0, WIDTH - BALL_RADIUS)
ball_y = 0
ball_speed = 5
score = 0

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start the webcam
cap = cv2.VideoCapture(0)

# Define the hand tracking function
def get_hand_position(frame):
    # Convert the frame to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the hand landmarks
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the position of the palm (center of the hand)
            palm_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH)
            palm_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT)
            
            return palm_x, palm_y
    return None, None

# Game loop
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get hand position (if available)
    hand_x, hand_y = get_hand_position(frame)
    
    if hand_x is not None and hand_y is not None:
        # Draw the hand position (palm)
        cv2.circle(frame, (hand_x, hand_y), 15, (0, 255, 0), -1)

    # Update the ball position
    ball_y += ball_speed

    # Check if the ball goes off the screen (game over condition)
    if ball_y > HEIGHT:
        cv2.putText(frame, "Game Over! Press 'q' to Quit", (250, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Hand Tracking Catch Ball", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit after game over
            break
        ball_y = 0
        ball_x = random.randint(0, WIDTH - BALL_RADIUS)  # Reset ball

    # Check if the ball is caught by the hand (simple distance check)
    if hand_x and hand_y:
        if (hand_x - BALL_RADIUS <= ball_x <= hand_x + PADDLE_WIDTH) and (hand_y - BALL_RADIUS <= ball_y <= hand_y + BALL_RADIUS):
            score += 1
            ball_y = 0  # Reset the ball position
            ball_x = random.randint(0, WIDTH - BALL_RADIUS)  # Randomize new ball position

    # Draw the ball (rectangle representing the falling ball)
    cv2.circle(frame, (ball_x, ball_y), BALL_RADIUS, (0, 0, 255), -1)
    
    # Draw the current score on the screen
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Hand Tracking Catch Ball", frame)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

# Release the webcam and close the game window
cap.release()
cv2.destroyAllWindows()
