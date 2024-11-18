from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Initialize YOLO model

    def detect_objects(self, frame):
        # Run inference on the frame
        results = self.model(frame)

        # Initialize lists for detected people and obstacles
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

        return people, obstacles
