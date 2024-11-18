from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 lightweight model

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting Btrolley detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w, _ = frame.shape
    frame_center = (w // 2, h // 2)

    # Run YOLO inference
    results = model(frame)

    # Initialize lists to hold bounding box data
    people = []
    obstacles = []

    # Separate detections into people and obstacles
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box
        class_id = int(detection.cls[0])  # Class ID
        confidence = detection.conf[0]  # Confidence score

        if confidence > 0.5:
            if class_id == 0:  # Class 0 is 'person' in COCO
                people.append((x1, y1, x2, y2))
            else:
                obstacles.append((x1, y1, x2, y2))

    # Find the person closest to the center
    closest_person = None
    min_distance = float('inf')

    for person in people:
        x1, y1, x2, y2 = person
        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = np.linalg.norm(np.array(person_center) - np.array(frame_center))
        if distance < min_distance:
            min_distance = distance
            closest_person = person

    # Draw bounding boxes
    for person in people:
        color = (0, 255, 0) if person == closest_person else (0, 0, 255)
        x1, y1, x2, y2 = person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Btrolley Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
