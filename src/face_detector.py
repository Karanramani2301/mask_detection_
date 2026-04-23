import cv2
import os

class FaceDetector:
    def __init__(self):
        # We use a robust OpenCV HaarCascade model. 
        # A more advanced alternative for angled profiles could be MTCNN,
        # but HaarCascades are drastically faster for real-time inference on moderate hardware.
        # It detects frontal faces primarily.
        
        # In opencv, the default haarcascade can be directly invoked via path
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            raise IOError("Unable to load the cascade classifier xml file!")

    def detect_faces(self, image_bgr):
        """
        Takes an OpenCV BGR image matrix, converts to grayscale, 
        and extracts the bounding boxes of all detected faces.
        Returns a list of (x, y, w, h) tuples.
        """
        # Grayscale simplifies the image array (removes color channels)
        # and vastly improves the algorithmic detection speeds.
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor determines how much the image size is reduced at each image scale
        # minNeighbors dictates how many neighbors each candidate rectangle should have
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    @staticmethod
    def crop_face(image_bgr, box):
        """
        Takes the image and crops it by the bounding box metrics.
        Cropping significantly improves deep learning accuracy because 
        the classifier doesn't get confused by the person's body or the background.
        """
        (x, y, w, h) = box
        # Provide some padding if it doesn't go out of bounds
        return image_bgr[y:y+h, x:x+w]
