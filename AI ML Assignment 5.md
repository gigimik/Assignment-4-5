SANGU
Course: AI and ML for Cybersecurity
Lecturer: Paata Gogishvili
Student: Grigol Mikadze
Assignment 5
DDoS Detection with a CNN
# Introduction
This guide describes how to transform network traffic records from the
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv file into grayscale images
and then train a Convolutional Neural Network (CNN) to detect DDoS attacks.
Why Convert Network Data to Images?
1. Pattern Learning: CNNs are strong at learning spatial patterns from images
2. Feature Structure: Mapping features into a grid can preserve relationships between values
3. Visual Inspection: Image representations make it easier to observe traffic behavior
4. Deep Learning Advantage: Enables computer-vision style learning for cybersecurity tasks
# Pipeline Architecture
CSV Data → Feature Extraction → Scaling/Normalization → Image Conversion → CNN Training → Grad-CAM Visualization
# Key Components
1. Data Preprocessing: Clean, validate, and scale network flow features
2. Image Generation: Convert each flow into a 9×9 grayscale image
3. CNN Model: Train a deep learning classifier for binary detection (Benign vs DDoS)
4. Grad-CAM: Produce heatmaps showing which regions influenced the model decision
# Required Libraries
1. numpy
2. pandas
3. tensorflow
4. opencv-python
5. scikit-learn
# Understanding the Dataset
Dataset: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
This file is part of the CICIDS2017 collection and includes traffic captures containing DDoS activity.

Dataset Structure
1. Rows: Each row corresponds to one network flow
2. Columns: ~80+ measured features describing the traffic behavior
3. Label Column: Contains the class label such as "BENIGN" or "DDoS"
# Sample Features
- Flow Duration
- Total Fwd Packets
- Total Backward Packets
- Flow Bytes/s
- Flow Packets/s
- Packet Length Mean/Std/Max/Min
- Protocol Type
- Port Numbers
- Flags Count
- And many more...
# 1. Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
Explanation:
1. Import libraries required for data processing, model training, and evaluation
2. cv2 (OpenCV) is used for image resizing and saving
3. tensorflow provides the CNN framework
4. scikit-learn is used for scaling, splitting, and reporting metrics
# 2. Configuration
DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
MODEL_PATH = "cnn_ddos_detector.keras"

IMG_SIZE = 9              # CNN input image size (9x9)
DISPLAY_SIZE = 224        # Size for saved images (visualization)
NUM_PIXELS = IMG_SIZE * IMG_SIZE

EPOCHS = 25
BATCH_SIZE = 512

# Output folders
os.makedirs("output/grayscale_images", exist_ok=True)
os.makedirs("output/gradcam_images", exist_ok=True)
Explanation:
1. IMG_SIZE = 9: Generates a 9×9 image (81 pixels) for each flow
2. DISPLAY_SIZE = 224: Upscales images to 224×224 for easier viewing
3. NUM_PIXELS = 81: Total pixel count in the working representation
4. EPOCHS = 25: Training passes over the dataset
5. BATCH_SIZE = 512: Samples per training step

Why 9×9 images?
A 9×9 grid holds exactly 81 values. The dataset contains around ~80 numeric features,
so it fits well with minimal padding or truncation.
# 3. Load Dataset
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Create binary labels:
# 1 = DDoS, 0 = Benign
y = df[" Label"].apply(lambda x: 1 if "DDoS" in x else 0).values

# Keep numeric features only
df = df.select_dtypes(include=[np.number])

# Handle NaN and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X = df.values
Explanation:
1. Load CSV: pd.read_csv() loads the dataset into a DataFrame
2. Extract Labels:
   - The label column includes values such as "BENIGN" or "DDoS"
   - Convert to binary integers: 1 for DDoS, 0 for Benign
3. Select Numeric Features:
   - Drop non-numeric columns (e.g., IPs/timestamps if present)
4. Handle Invalid Values:
   - Replace inf/-inf with NaN (often caused by division-by-zero features)
   - Fill NaN with 0 to keep the pipeline stable
5. Build Feature Matrix:
   - X becomes a numeric 2D array of shape (samples, features)
# 4. Feature Normalization
print("[INFO] Normalizing features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Why Normalize?
1. Traffic features can have very different value ranges (e.g., counts vs ports vs durations)
2. MinMaxScaler maps values into [0, 1]:
   scaled = (value - min) / (max - min)

This helps:
- stabilize training
- make pixel values consistent for image conversion
- prevent large-range features from dominating learning
# 5. Convert Flows to Grayscale Images
def flow_to_image(flow):
    """
    Convert a single network flow (feature vector)
    into a 9x9 grayscale image.
    """
    if len(flow) < NUM_PIXELS:
        flow = np.pad(flow, (0, NUM_PIXELS - len(flow)))
    else:
        flow = flow[:NUM_PIXELS]

    return flow.reshape(IMG_SIZE, IMG_SIZE)

X_images = np.array([flow_to_image(flow) for flow in X_scaled])
X_images = X_images[..., np.newaxis]  # Add channel dimension (grayscale)
Explanation:
1. Each flow is a 1D vector of feature values
2. The vector is padded or truncated to exactly 81 values
3. The 81 values are reshaped into a 9×9 grid (image)
4. A channel dimension is added so CNN input becomes (9, 9, 1)

Result:
X_images.shape = (N, 9, 9, 1)
# 6. Save Grayscale Flow Images
print("[INFO] Saving grayscale flow images...")

for i in range(min(300, len(X_images))):
    label = "ddos" if y[i] == 1 else "benign"

    img = np.uint8(255 * X_images[i].squeeze())

    # Upscale for visualization
    img_resized = cv2.resize(
        img,
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        f"output/grayscale_images/{label}_{i}.png",
        img_resized
    )
Explanation:
1. Save a sample of images (first 300) for inspection
2. Convert image pixels from [0,1] float to [0,255] uint8
3. Upscale 9×9 to 224×224 using nearest-neighbor to preserve block structure
4. Write files to output/grayscale_images/
# 7. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_images,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Explanation:
- 80% training, 20% testing
- stratify=y preserves the same class ratio in both sets
- random_state ensures reproducible splits
# 8. Build CNN Model
print("[INFO] Building CNN model...")

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1")(inputs)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2")(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs, name="CNN_DDoS_Detector")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
Explanation:
- Two Conv2D layers learn local patterns within the 9×9 grid
- BatchNormalization improves stability and convergence
- Flatten converts feature maps to a vector
- Dense + Dropout learns higher-level combinations and reduces overfitting
- Sigmoid output produces probability of DDoS (binary classification)
# 9. Train CNN Model
print("[INFO] Training CNN...")
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# Save trained model
model.save(MODEL_PATH)
print(f"[INFO] Model saved as {MODEL_PATH}")
Explanation:
- validation_split=0.2 uses 20% of training data for validation checks
- Model is saved for later reuse/deployment
# 10. Evaluate Model
print("[INFO] Evaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
Explanation:
- Produces precision, recall, and F1-score for each class
- High values indicate effective separation between benign and DDoS flows
# 11. GRAD-CAM Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap for a single image.
    """
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap /= max_val

    return heatmap.numpy()
Explanation:
Grad-CAM highlights which areas of the feature-image contributed most to the model’s prediction.
Brighter regions indicate higher importance.
# 12. Save GRAD-CAM Overlay Images
print("[INFO] Generating Grad-CAM images...")

LAST_CONV_LAYER = "conv_2"

for i in range(50):
    img = X_test[i]
    img_input = np.expand_dims(img, axis=0)

    heatmap = make_gradcam_heatmap(
        img_input,
        model,
        LAST_CONV_LAYER
    )

    # Resize heatmap for visualization
    heatmap = cv2.resize(
        heatmap,
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize original grayscale image
    original = np.uint8(255 * img.squeeze())
    original = cv2.resize(
        original,
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(
        original, 0.6,
        heatmap, 0.4,
        0
    )

    label = "ddos" if y_test[i] == 1 else "benign"
    cv2.imwrite(
        f"output/gradcam_images/gradcam_{label}_{i}.png",
        overlay
    )

print("[SUCCESS] Grayscale images, CNN model, and Grad-CAM heatmaps generated.")
Explanation:
- Grad-CAM heatmaps are resized and colorized for readability
- Heatmaps are overlaid on the grayscale image
- Results are saved into output/gradcam_images/
# Results and Interpretation
Expected Outputs
1. Grayscale Images
output/grayscale_images/
- benign_*.png (traffic pattern samples)
- ddos_*.png   (attack pattern samples)

2. Trained Model
cnn_ddos_detector.keras

3. Grad-CAM Images
output/gradcam_images/
- gradcam_benign_*.png
- gradcam_ddos_*.png

Interpretation:
- Hot (red/yellow) areas = regions most influential to the model
- Cooler (blue) areas = less relevant regions
# Usage Instructions
Step 1: Prepare Environment
- Install required libraries
- Create a project folder

Step 2: Download Dataset
- Place Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv in the project directory

Step 3: Run the Code
- Execute the notebook/script

Step 4: Monitor Training
- Track loss and accuracy per epoch

Step 5: Review Results
- output/grayscale_images/: view traffic patterns
- output/gradcam_images/: interpret model attention
- cnn_ddos_detector.keras: saved trained model
# Conclusion
This guide demonstrates how to:
1. Convert network traffic CSV records into grayscale images
2. Train a CNN to recognize DDoS-related patterns
3. Apply Grad-CAM to interpret important regions used by the model
4. Achieve high accuracy for DDoS detection