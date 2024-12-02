import cv2
from pyzbar.pyzbar import decode
import numpy as np

def scan_qr_code_and_save_to_file(output_file):
    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    print("Point the camera at a QR code. Press 'q' to quit.")
    
    scanned_data = set()  # To store unique QR code data
    
    try:
        with open(output_file, 'a') as file:
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                decoded_objects = decode(frame)

                # Process each detected QR code
                for obj in decoded_objects:
                    # Get the bounding box coordinates
                    points = obj.polygon
                    if len(points) > 4:  # Convex hull for complex polygons
                        hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                        points = hull

                    # Draw the polygon on the frame
                    n = len(points)
                    for j in range(n):
                        pt1 = tuple(points[j].astype(int))
                        pt2 = tuple(points[(j + 1) % n].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                    # Decode the QR code data
                    qr_data = obj.data.decode("utf-8").strip()
                    if qr_data and qr_data not in scanned_data:
                        print(f"QR Code Data: {qr_data}")
                        scanned_data.add(qr_data)  # Add to the set of scanned data
                        file.write(qr_data + '\n')  # Save to the file

                # Show the frame with bounding boxes
                cv2.imshow("QR Code Scanner", frame)

                # Exit loop while pressing q 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

# Run the function
scan_qr_code_and_save_to_file("scanned_qr_codes.txt")

