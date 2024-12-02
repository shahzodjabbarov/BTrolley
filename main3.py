from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path  
script_dir = Path(__file__).resolve().parent
video_path = script_dir / 'video1.MOV'
# Initialize YOLO model
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

# Trackers dictionary
tracked_objects = {}
next_object_id = 2  # Start from 2 because ID 1 will be the subject
subject_id = None
frame_center = None
init_phase = True  # To check if the subject is being initialized
init_frames = 0    # Frame count for initialization

def calculate_cost_matrix(tracked_objects, detections):
    num_tracks = len(tracked_objects)
    num_detections = len(detections)

    cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

    for i, (track_id, track_data) in enumerate(tracked_objects.items()):
        for j, detection in enumerate(detections):
            bbox = detection.xyxy[0].cpu().numpy()
            iou = calculate_iou(track_data['bbox'], bbox)
            cost_matrix[i, j] = 1 - iou  # Cost is inverse of IoU (lower is better)

    return cost_matrix

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def get_center_distance(bbox, center):
    x1, y1, x2, y2 = bbox
    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return np.linalg.norm(np.array(bbox_center) - np.array(center))

def update_tracks(tracked_objects, matches, unmatched_tracks, unmatched_detections, detections):
    """
    Update existing tracks and create new tracks for unmatched detections.
    """
    global next_object_id, subject_id

    # Update matched tracks
    for track_idx, detection_idx in matches:
        track_id = list(tracked_objects.keys())[track_idx]
        bbox = detections[detection_idx].xyxy[0].cpu().numpy()
        tracked_objects[track_id]['bbox'] = bbox
        tracked_objects[track_id]['age'] = 0

    # Create new tracks for unmatched detections
    for detection_idx in unmatched_detections:
        bbox = detections[detection_idx].xyxy[0].cpu().numpy()
        tracked_objects[next_object_id] = {'bbox': bbox, 'age': 0}
        next_object_id += 1

    # Remove unmatched tracks immediately
    unmatched_keys = [list(tracked_objects.keys())[idx] for idx in unmatched_tracks]
    for track_id in unmatched_keys:
        if track_id != subject_id:  # Ensure we do not delete the subject's track
            del tracked_objects[track_id]



# Processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_center = (width / 2, height / 2)

    if init_phase:
        init_frames += 1
        cv2.putText(frame, "Please stand in the middle of the frame.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if init_frames >= 150:  # 5 seconds at 30 FPS
            results = model(frame, classes=[0])  # Detect persons
            detections = results[0].boxes if results[0].boxes else []

            if len(detections) > 0:
                # Find the closest person to the center
                closest_detection = min(detections, key=lambda d: get_center_distance(d.xyxy[0].cpu().numpy(), frame_center))
                bbox = closest_detection.xyxy[0].cpu().numpy()
                subject_id = 1
                tracked_objects[subject_id] = {'bbox': bbox, 'age': 0}
                init_phase = False
            else:
                continue

    else:
        results = model(frame, classes=[0])  # Detect persons
        detections = results[0].boxes if results[0].boxes else []

        if len(tracked_objects) > 0 and len(detections) > 0:
            cost_matrix = calculate_cost_matrix(tracked_objects, detections)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = list(zip(row_ind, col_ind))
            unmatched_tracks = set(range(len(tracked_objects))) - set(row_ind)
            unmatched_detections = set(range(len(detections))) - set(col_ind)
            update_tracks(tracked_objects, matches, unmatched_tracks, unmatched_detections, detections)
        else:
            for detection in detections:
                bbox = detection.xyxy[0].cpu().numpy()
                tracked_objects[next_object_id] = {'bbox': bbox, 'age': 0}
                next_object_id += 1

    # Draw tracked objects
    for track_id, track_data in tracked_objects.items():
        bbox = track_data['bbox']
        color = (0, 255, 0) if track_id == subject_id else (0, 0, 255)  # Green for subject, red for others
        label = f"Person {track_id}" if track_id != subject_id else "Subject"
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
