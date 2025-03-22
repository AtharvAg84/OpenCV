import cv2
import mediapipe as mp
import numpy as np

# Set up the game window dimensions
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 60  # Increased size of cells for better visibility

# Set the maze dimensions
MAZE_WIDTH = 9  # Number of columns
MAZE_HEIGHT = 9  # Number of rows

# New strict maze layout with a guaranteed path
maze = [
    [1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1]
]


# Start and end positions
start_pos = (1, 1)  # Start at maze[1][1]
end_pos = (7, 7)    # End at maze[7][7]

# Initial player position
player_pos = list(start_pos)

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start the webcam (initialize cap)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Draw the maze on the screen
def draw_maze(frame):
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            color = (0, 0, 0) if maze[y][x] == 1 else (255, 255, 255)  # Walls are black, paths are white
            cv2.rectangle(frame, (x * CELL_SIZE, y * CELL_SIZE),
                          ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), color, -1)

# Function to get hand position and gesture
def get_hand_position_and_gesture(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the wrist landmark position (the base of the hand)
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH)
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT)
            
            # Use hand gestures to control movement
            finger_tips = [
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
            ]
            
            # Gesture detection logic
            is_open_hand = all(finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y for finger in finger_tips)
            is_closed_fist = all(finger.y > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y for finger in finger_tips)
            
            # Check for left or right thumb pointing gestures
            is_thumb_left = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            is_thumb_right = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

            return wrist_x, wrist_y, is_open_hand, is_closed_fist, is_thumb_left, is_thumb_right
    return None, None, False, False, False, False

# Game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Draw maze
    draw_maze(frame)

    # Get hand position and gesture
    hand_x, hand_y, is_open_hand, is_closed_fist, is_thumb_left, is_thumb_right = get_hand_position_and_gesture(frame)

    # Debugging the gesture recognition
    if is_open_hand:
        cv2.putText(frame, "Move Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif is_closed_fist:
        cv2.putText(frame, "Move Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif is_thumb_left:
        cv2.putText(frame, "Move Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif is_thumb_right:
        cv2.putText(frame, "Move Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Move player based on gestures
    if is_open_hand:
        # Move Up
        if player_pos[1] > 0 and maze[player_pos[1] - 1][player_pos[0]] == 0:
            player_pos[1] -= 1
    elif is_closed_fist:
        # Move Down
        if player_pos[1] < MAZE_HEIGHT - 1 and maze[player_pos[1] + 1][player_pos[0]] == 0:
            player_pos[1] += 1
    elif is_thumb_left:
        # Move Left
        if player_pos[0] > 0 and maze[player_pos[1]][player_pos[0] - 1] == 0:
            player_pos[0] -= 1
    elif is_thumb_right:
        # Move Right
        if player_pos[0] < MAZE_WIDTH - 1 and maze[player_pos[1]][player_pos[0] + 1] == 0:
            player_pos[0] += 1

    # Draw player on the maze
    cv2.circle(frame, (player_pos[0] * CELL_SIZE + CELL_SIZE // 2, player_pos[1] * CELL_SIZE + CELL_SIZE // 2),
               CELL_SIZE // 3, (0, 0, 255), -1)

    # Check if player reached the end
    if player_pos == list(end_pos):
        cv2.putText(frame, "You Win!", (WIDTH // 2 - 100, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("Gesture Maze Solver", frame)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

# Release the webcam and close the game window
cap.release()
cv2.destroyAllWindows()
