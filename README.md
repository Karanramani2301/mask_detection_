# Mask Detection and Face Safety Compliance Checker (Advanced)

A comprehensive Computer Vision and Deep Learning system designed to track, classify, and enforce face mask compliance in real-time. Built specifically to handle three classes of compliance matching industrial standards: **Mask Worn Properly, Mask Worn Improperly, and No Mask**.

## 📌 Project Overview
This project detects human faces from a static image or a real-time webcam feed. Each detected face is then classified using an optimized **MobileNetV2** Transfer Learning architecture. An intelligent **Compliance Score (0-100)** is calculated based on class labels and confidence metrics to evaluate standard safety thresholds.

## 📊 Dataset Details
The model was trained on the open-source Kaggle Face Mask dataset with the following preprocessing applied:
- **Image Resizing**: Scaled uniformly to `224x224` to fit ImageNet specifications.
- **Normalization**: Pixel values rescaled to `[-1, 1]` for MobileNetV2 inputs.
- **Data Augmentation**: To prevent overfitting on constrained real-world angles, we employed Image Augmentation:
  - Random Zooming (up to 20%)
  - Rotation (up to 20 degrees)
  - Horizontal Flips
  - Height & Width shifting

**Why Data Augmentation?** It improves the model's robustness by artificially expanding the training dataset, giving the neural network exposed variations of angles, lighting, and placement, reducing the chance of memorizing particular images (overfitting).

## 🧠 Model Architecture & Explanation
We chose **Transfer Learning with MobileNetV2** because it is exceptionally lightweight and optimized for edge devices and real-time inference formats, making it ideal for webcam streams without needing massive GPUs.
- **Base Model**: Pre-trained on ImageNet.
- **Head**: Standard GlobalAveragePooling2D, followed by a Dropout layer (0.5 rate) to prevent overfitting, into a Dense layer with `Softmax` output for the 3 distinct classes.
- **Rough Parameters**: ~2.3 Million parameters.

Signs of over-fitting (like divergence between training loss and validation loss) were managed by aggressive Model Checkpointing, Early Stopping callbacks, and robust dropout.

## 🎯 Compliance Scoring Logic
Each classification outputs a Softmax Confidence Level. We use this to compute a granular grade:

- **Properly Worn**: `Score = Confidence % * 100`
- **Improperly Worn**: `Score = Confidence % * 60`
- **No Mask**: `Score = Confidence % * 20`

If the confidence is uncertain (`<0.6`), a penalty drop of `-20%` is added to clearly highlight ambiguous cases.

**Final Status Engine:**
1. **✅ SAFE**: `Score >= 80`
2. **⚠️ UNCERTAIN**: `Confidence < 60% OR Score between 50 and 80`
3. **⛔ VIOLATION**: `Score < 50`

## 🔎 Face Detection Module
Uses robust Pre-trained cascade classification (`haarcascade_frontalface_default.xml`):
- Dynamic extraction captures just the bounded box.
- **Cropping faces**: Extensively improves classification accuracy by excluding distractive background elements (clothing, trees, lighting) matching only the facial geometry features to our Deep Learning model.
- *Limitation*: Can struggle strongly from extreme profile side-angles compared to highly complex Deep Neural Nets (like MTCNN).

## 📈 Accuracy Results & Matrix
The pipeline achieved robust Validation metrics across proper cross-entropy splits (`>90%` Acc).
Refer to `models/training_history.png` and `models/confusion_matrix.png` for comprehensive visualizations!

### Confusion Matrix Explanation
The confusion matrix charts the Actual label versus Predicted label on our test-set holdout. A perfect model would show bright high frequencies aligned diagonally. Some bleeding occasionally occurs distinguishing 'improper' from 'proper', mostly due to tricky nose-bridge shadows.

## 🚧 Challenges Faced 
1. Real-time inference speeds dropping when numerous faces overlap.
2. Distinctive 'improperly' worn styles are highly asymmetrical in the dataset compared to the other classes.
3. Hardware constraints ensuring Streamlit can sync properly with concurrent video processing frames.

## 🚀 Future Improvements
1. Integrating audio (Sound Alarms) for repeated compliance violations.
2. Expanding out to SSD/MTCNN detection arrays for profile-view face capturing.

---
### To Run Locally
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
