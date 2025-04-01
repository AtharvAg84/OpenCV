import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set up the drawing canvas
canvas_width = 640
canvas_height = 480669
drawing_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White canvas
brush_color = (0, 0, 0)  # Initial color (black)
brush_thickness = 5  # Initial brush thickness

# Set the webcam
cap = cv2.VideoCapture(0)

# Initialize the previous finger positions for dynamic brush thickness
prev_thumb_index_distance = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(rgb_frame)

    # Initialize variables for drawing
    drawing = False
    hand_positions = []

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb and index finger landmarks (for brush thickness)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_index_distance = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
            )

            # Dynamic Brush Thickness (distance between thumb and index fingers)
            if thumb_index_distance > 0.1:
                brush_thickness = int(thumb_index_distance * 50)

            # Handle gestures and drawing actions
            if thumb_index_distance > 0.05:
                # Check if the pinky finger is raised for Selection Mode
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                if pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                    drawing = False  # Prevent accidental drawing (Selection Mode)
                else:
                    drawing = True  # Start drawing

            if drawing:
                # Draw on the canvas (based on hand landmarks)
                for i in range(1, len(hand_landmarks.landmark)):
                    x1, y1 = int(hand_landmarks.landmark[i].x * frame.shape[1]), int(hand_landmarks.landmark[i].y * frame.shape[0])
                    cv2.circle(frame, (x1, y1), brush_thickness, brush_color, -1)
                    hand_positions.append((x1, y1))

    # Display a temporary highlight where the hand will draw
    if hand_positions:
        cv2.polylines(frame, [np.array(hand_positions)], isClosed=False, color=brush_color, thickness=brush_thickness)

    # Now draw on the canvas (not the frame), which will be displayed in the final result
    if drawing:
        for i in range(1, len(hand_landmarks.landmark)):
            x1, y1 = int(hand_landmarks.landmark[i].x * canvas_width), int(hand_landmarks.landmark[i].y * canvas_height)
            cv2.circle(drawing_canvas, (x1, y1), brush_thickness, brush_color, -1)

    # Show the drawing canvas on the window
    cv2.imshow("Hand Gesture Drawing App", frame)

    # Display the drawing canvas as an overlay or in a separate window
    cv2.imshow("Drawing Canvas", drawing_canvas)

    # Close the drawing canvas with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
