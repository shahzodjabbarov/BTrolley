from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
# video_path = 0 # webcam
video_path = r'C:\Users\chawt\Desktop\IoT\Code\test_video\homeplus_shot2.mp4'
cap = cv2.VideoCapture(video_path)


# store the bounding boxes of the first detected object in previous frame
previous_sizes = {} # a dict has been used to associate IDs with corresponding sizes


ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # get the size of the bounding boxes
        for box in results[0].boxes:
            # get the coordinates of the first detected object's bounding boxes
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width, height = x2 - x1, y2 - y1
            # object id to make sure that bounding boxes of the same object are compared later
            object_id = int(box.id[0])

            # TO SHOW THE OBJECT NAME instead of just an id
            # get the class id and map it to the class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # map class ID to class name


            # Check if the object has been seen before (ID already in previous_sizes)
            if object_id in previous_sizes:
                last_size = previous_sizes[object_id]
                width_diff = abs(width - last_size[0])
                height_diff = abs(height - last_size[1])

                # If the difference in size exceeds a threshold, it's a significant change
                if width_diff > 20 or height_diff > 10:
                    print(
                        f"Significant size change detected for {class_name} (Object ID {object_id})! "
                        f"Width diff: {width_diff}, Height diff: {height_diff}"
                    )
            # Update the previous size for this object ID
            previous_sizes[object_id] = (width, height)



        # plot results
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    # add error handling
    else:
        print("Error loading the video file or the webcam is not supported")
        exit()
