# BUDDY-TROLLEY - Smart Trolley System

The **Smart Trolley System** is a computer vision-based project designed to track and follow a specific person in real-time. This innovative solution leverages object detection and tracking technologies to enhance the shopping experience, making it easier for customers to shop hands-free while the trolley follows them.

https://github.com/user-attachments/assets/dc21c6f8-4179-4b55-af62-695ab0755f8f

https://github.com/shahzodjabbarov/BTrolley/blob/main/videos/output1.gif
---

## 🚀 Features
- **Person Detection:** Utilizes the YOLOv8 model to detect people in the video feed.
- **Person Tracking:** Tracks the selected subject using the Intersection over Union (IoU) metric and the Hungarian Algorithm for optimal matching.
- **Subject Identification:** Automatically selects the closest person to the center of the frame as the subject after a 5-second initialization phase.
- **Re-identification (Re-ID):** Reassociates lost or unmatched detections to maintain subject tracking.
- **Optimized Performance:** Skips frames and uses efficient algorithms to ensure smooth real-time processing.    
### Trolley movement     
- The movement of the trolley is dynamically controlled based on the subject's location within the video frame. The functionality includes the following features:
- **Zone-based Tracking:** The frame is divided into three distinct zones: left, center, and right. The subject's position within these zones determines the trolley's movement. If the subject moves into the left or right zone, the trolley adjusts its direction to follow. When the subject is in the center zone, the trolley continues straight.
- **Proximity Control:** To ensure safety, the system monitors the distance between the trolley and the subject. If the subject moves too close (within a predefined distance threshold), the trolley automatically stops to prevent collisions.

---

## 🛠️ Technologies Used
- **[YOLOv8](https://github.com/ultralytics/ultralytics):** For real-time object detection.
- **OpenCV:** For video processing and drawing bounding boxes.
- **NumPy:** For numerical computations.
- **Scipy (Hungarian Algorithm):** For solving the assignment problem in tracking.
- **Python 3.9+**

---

## 📂 File Structure
```plaintext
├── smart_trolley/
│   ├── main4.py                #(https://github.com/shahzodjabbarov/BTrolley/blob/39084d8ed55655e0513dd949bdbbe1b84f2c3de5/main4.py)
│   ├── requirements.txt       # List of dependencies
│   ├── video3.mp4             # (https://drive.google.com/file/d/1HSQxuAdFbO4hU6Nfb92ywRRq55xgg1R8/view?usp=share_link)
│   ├── README.md              # Project documentation
└── ...


🖥️ How It Works

Initialization Phase:
The system identifies the closest person to the frame center after 5 seconds and marks them as the subject.
Detection & Tracking:
Using YOLOv8, the system detects people in each frame and tracks the subject based on IoU.
Re-Identification:
If the subject is lost, the system attempts to reidentify them among new detections.
Visualization:
Bounding boxes are drawn around the subject (green) and other detected people (red).
📋 Requirements

Python 3.9+
Install dependencies with:
pip install -r requirements.txt
Content of requirements.txt:
ultralytics
opencv-python
numpy
scipy
▶️ Running the Project

Clone the repository:
git clone https://github.com/your-username/smart-trolley.git
cd smart-trolley
Add a test video file named video3.mp4 in the project directory.
Run the main script:
python main.py
Press q to exit the video stream.
📖 Explanation of Key Code Sections

Person Detection: YOLOv8 is used to detect people in each frame, filtering detections by confidence and class ID.
Tracking: The Hungarian Algorithm minimizes tracking errors by pairing existing tracked objects with new detections based on IoU.
Subject Selection: Automatically selects the subject closest to the frame center during initialization.
Frame Skipping: Processes every 4th frame to balance performance and accuracy.
📈 Performance Notes

Frames per Second (FPS): The system processes approximately X FPS depending on the hardware.
Optimization Strategies: Frame skipping and IoU-based cost minimization ensure real-time processing.
* Google Drive Folder For Videos: https://drive.google.com/drive/folders/1xQtF1-Vwg8Pz7xHE3nHaSexEY7Hudigg?usp=sharing
* Google Drive Folder: https://drive.google.com/drive/folders/1iK7ix4OvUB72aQc94lEAcuEeP7qXb_QQ
