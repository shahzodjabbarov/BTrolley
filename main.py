from detection import ObjectDetector
from depth_estimation import DepthEstimatorWrapper
import cv2
import numpy as np

# Load the YOLO model and Depth Estimator
object_detector = ObjectDetector('yolov8n.pt')  # YOLOv8 lightweight model
depth_estimator = DepthEstimatorWrapper()  # Use the default pipeline from transformers

# Calibration: SF = known_real_distance / depth_value( value detected by the model )
scale_factor = 0.2685 # adjusted after comparing real-time value and default detected value

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
    people, obstacles = object_detector.detect_objects(frame)

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

    # Draw bounding boxes for people
    for person in people:
        color = (0, 255, 0) if person == closest_person else (0, 0, 255)
        x1, y1, x2, y2 = person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw bounding boxes for obstacles
    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Apply depth estimation using MiDaS for detected obstacles
    depth_map = depth_estimator.estimate_depth(frame)
    depth_map_resized = cv2.resize(depth_map, (w, h))  # Resize depth map to match the frame size

    # Calculate depth values for obstacles and person
    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle
        # Extract the depth at the center of the bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        try:
            depth_value = depth_map_resized[center_y, center_x] * scale_factor  # Apply scale factor
        except IndexError:
            depth_value = 0.0  # If out of bounds, set depth to 0

        # Display depth information for obstacles
        """ the depth information is a little bit different from real world measurement, 
         to avoid confusion this has been commented out, 
         remove # to recalculate the scaling factor when necessary"""
        #cv2.putText(frame, f"Depth: {depth_value:.2f}m", (x1, y1 - 10),
         #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure closest person and obstacle are defined before calculating distance
    if closest_person is not None:
        # Calculate the center of the bounding boxes for closest person
        person_center_x, person_center_y = (closest_person[0] + closest_person[2]) // 2, (closest_person[1] + closest_person[3]) // 2

        # Get the depth value at the center of the closest person
        person_depth_value = depth_map_resized[person_center_y, person_center_x] * scale_factor

        # Calculate distance to each obstacle
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle
            # Calculate the center of the obstacle
            object_center_x, object_center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Get the depth value at the center of the object (obstacle)
            object_depth_value = depth_map_resized[object_center_y, object_center_x] * scale_factor

            # Calculate the distance (in this case, relative depth difference)
            distance = abs(person_depth_value - object_depth_value)

            # Display the distance on the frame (optional)
            cv2.putText(frame, f"Distance: {distance:.2f}cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Btrolley Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
