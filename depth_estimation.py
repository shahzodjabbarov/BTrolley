from transformers import pipeline
import cv2
from PIL import Image
import numpy as np


class DepthEstimatorWrapper:
    def __init__(self):
        # Initialize the depth estimation pipeline from Hugging Face's transformers
        self.pipe = pipeline(task="depth-estimation", model="Intel/dpt-swinv2-tiny-256")

    def estimate_depth(self, frame):
        # Convert the OpenCV frame (BGR) to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(image)

        # Ensure the image is in RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Pass the PIL image to the pipeline
        result = self.pipe(pil_image)

        # Convert the depth map to a NumPy array (if it's not already)
        depth_map = np.array(result["depth"])

        # Return the depth map as a NumPy array
        return depth_map
