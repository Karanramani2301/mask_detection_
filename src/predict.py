import cv2
import numpy as np
import tensorflow as tf
from src.face_detector import FaceDetector
from src.compliance_score import calculate_compliance

class MaskPredictor:
    def __init__(self, model_path="models/mask_detector.h5"):
        self.detector = FaceDetector()
        print(f"[INFO] Loading deep learning model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"[ERROR] Could not load model. Ensure it exists. {e}")
            self.model = None

    def process_frame(self, frame):
        """
        Main pipeline logic:
        1. Takes frame
        2. Detects faces via Pipeline
        3. Iterates over faces, crops them, pre-processes
        4. Predicts classes
        5. Computes Compliance
        6. Draws Bounding Boxes and Layouts
        Returns annotated bounding frame
        """
        orig_frame = frame.copy()
        
        # 1. Detect faces
        faces = self.detector.detect_faces(frame)
        
        if len(faces) == 0:
            return orig_frame
            
        faces_cropped = []
        locs = []
        
        # 2. Extract bounding boxes
        for (x, y, w, h) in faces:
            face_img = self.detector.crop_face(frame, (x, y, w, h))
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue
                
            # Preprocess to feed to MobileNetV2
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = tf.keras.preprocessing.image.img_to_array(face_img)
            face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
            
            faces_cropped.append(face_img)
            locs.append((x, y, w, h))
            
        if len(faces_cropped) > 0 and self.model is not None:
            faces_cropped = np.array(faces_cropped, dtype="float32")
            
            # Predict in batch
            preds = self.model.predict(faces_cropped, batch_size=32, verbose=0)
            
            for (box, pred) in zip(locs, preds):
                (x, y, w, h) = box
                
                # Fetch intelligence analysis
                analysis = calculate_compliance(pred)
                
                label = analysis['label']
                b_status = analysis['status']
                b_score = analysis['score']
                
                # Dynamic Bounding Box Colors
                if b_status == "SAFE":
                    color = (0, 255, 0) # Green
                elif b_status == "VIOLATION":
                    color = (0, 0, 255) # Red
                else: 
                    color = (0, 255, 255) # Yellow/Orange
                    
                # Add bounding box
                cv2.rectangle(orig_frame, (x, y), (x + w, y + h), color, 2)
                
                # Add Text
                text = f"{label}: {analysis['confidence']}% | Score: {b_score} | {b_status}"
                
                # Background rect for text readability
                label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                y_text = max(y, label_size[1] + 10)
                
                # Optional: Overlay rectangle background for the text
                cv2.rectangle(orig_frame, (x, y_text - label_size[1] - 10), 
                             (x + label_size[0], y_text + base_line - 10), color, cv2.FILLED)
                
                cv2.putText(orig_frame, text, (x, y_text - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                            
        return orig_frame
