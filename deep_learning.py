from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with 'yolov11n.pt' for YOLOv11

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLO Detection', annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
