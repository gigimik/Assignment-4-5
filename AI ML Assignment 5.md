```python
SANGU
```


```python
Course: AI and ML for Cybersecurity
Lecturer: Paata Gogishvili
```


```python
Student: Ana Margvelashvili
```


```python
Assignment 5
```


```python
DDoS Detection Using CNN
```


```python
# Introduction
This guide explains how to convert network traffic data from the Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv 
file into grayscale images and use a Convolutional Neural Network (CNN) to detect DDoS attacks.
```


```python
Why Convert Network Data to Images?
1. Pattern Recognition: CNNs excel at detecting spatial patterns in images
2. Feature Relationships: Converting flows to images preserves relationships between features
3. Visual Analysis: Images allow us to visually inspect attack patterns
4. Deep Learning Power: Leverage powerful computer vision techniques for network security
```


```python
# Pipeline Architecture
CSV Data → Feature Extraction → Normalization → Image Conversion → CNN Training → Grad-CAM Analysis
```


```python
# Key Components
1. Data Preprocessing: Clean and normalize network traffic features
2. Image Generation: Convert each network flow into a 9×9 grayscale image
3. CNN Model: Train a deep learning model for binary classification
4. Grad-CAM: Generate heatmaps showing which parts of the image influenced predictions
```


```python
# Required Libraries
1. numpy
2. pandas
3. tensorflow
4. opencv-python
5. scikit-learn
```


```python
# Understanding the Dataset
Dataset: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
This dataset is part of the CICIDS2017 dataset containing network traffic captured during DDoS attacks.

Dataset Structure
1. Rows: Each row represents a single network flow
2. Columns: ~80+ features describing the network traffic
3. Label Column: Contains "BENIGN" or "DDoS" classification
```


```python
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
```


```python
# 1. Import Libraries
```


```python
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
```


```python
Explanation:
1. Import essential libraries for data processing, machine learning, and visualization
2. cv2 (OpenCV) handles image operations
3. tensorflow provides the deep learning framework
4. sklearn tools help with preprocessing and evaluation
```


```python
# 2. Configuration
```


```python
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
```


```python
Explanation:
1. IMG_SIZE = 9: Creates a 9×9 = 81 pixel image. This is small but efficient for training
2. DISPLAY_SIZE = 224: Upscales images to 224×224 for better visualization (human viewing)
3. NUM_PIXELS = 81: Total number of pixels in our working image
4. EPOCHS = 25: Number of complete passes through the training data
5. BATCH_SIZE = 512: Number of samples processed before updating model weights

Why 9×9 images?
A 9×9 image requires exactly 81 features. The dataset has ~80 features, which fits perfectly. 
Larger images would require more features or padding with zeros.
```


```python
# 3. Load Dataset
```


```python
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Create binary labels
# 1 = DDoS, 0 = Benign
y = df[" Label"].apply(lambda x: 1 if "DDoS" in x else 0).values

# Keep numeric features only
df = df.select_dtypes(include=[np.number])

# Handle NaN and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X = df.values
```

    [INFO] Loading dataset...
    


```python
Explanation:
1. Load CSV: pd.read_csv() reads the entire dataset into a DataFrame
2. Extract Labels:
- The label column contains strings like "BENIGN" or "DDoS"
- Convert to binary: 1 for DDoS, 0 for Benign
- .values converts pandas Series to numpy array
3. Select Numeric Features:
- Remove non-numeric columns (like IP addresses, timestamps)
- Only numeric features can be used in calculations
4. Handle Invalid Values:
- Network data often has infinity values (division by zero)
- Replace inf and -inf with NaN
- Fill NaN with 0 to avoid errors
5. Create Feature Matrix: X now contains all numeric features as a 2D array

Shape Example:
X.shape = (225745, 78)  # 225,745 flows × 78 features
y.shape = (225745,)      # 225,745 labels
```


```python
# 4. Feature Normalization
```


```python
print("[INFO] Normalizing features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

    [INFO] Normalizing features...
    


```python
Why Normalize?
1. Network features have vastly different ranges:
- Packet count: 0 to 1,000,000
- Port number: 0 to 65,535
- Duration: 0 to 120,000,000 microseconds
2. MinMaxScaler transforms all features to [0, 1] range:
scaled_value = (value - min) / (max - min)

Example:
Original: [10, 1000, 50000]
Scaled:   [0.01, 0.12, 0.85]  (approximately)

This ensures:
- All features contribute equally
- CNN can learn patterns effectively
- Pixel values fit naturally in [0, 1] range
```


```python
# 5. Convert Flows - Grayscale Images
```


```python
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
X_images = X_images[..., np.newaxis]  # Add channel dimension
```


```python
Explanation:
This is the core concept of this approach!

How It Works:
Step 1: Start with a Flow Vector
# Example: A single network flow with 78 features
flow = [0.23, 0.81, 0.45, 0.12, ..., 0.67]  # 78 values
Each value represents a normalized network feature (packet size, duration, etc.)

Step 2: Pad or Truncate to 81 Values
# If flow has < 81 features: pad with zeros
if len(flow) < 81:
    flow = [0.23, 0.81, ..., 0.67, 0.00, 0.00, 0.00]  # Now 81 values

# If flow has > 81 features: take first 81
else:
    flow = flow[:81]

Step 3: Reshape to 9×9 Grid
image = flow.reshape(9, 9)

Visual Representation:
1D Vector (81 values):
[0.23, 0.81, 0.45, 0.12, 0.56, 0.89, 0.34, 0.67, 0.91, ...]

↓ Reshape to 9×9 ↓

2D Image (9×9 matrix):
[[0.23, 0.81, 0.45, 0.12, 0.56, 0.89, 0.34, 0.67, 0.91],
 [0.15, 0.78, 0.34, 0.92, 0.11, 0.56, 0.88, 0.23, 0.45],
 [0.67, 0.34, 0.89, 0.12, 0.78, 0.45, 0.91, 0.23, 0.56],
 [...],
 [0.34, 0.78, 0.23, 0.91, 0.45, 0.12, 0.67, 0.89, 0.34]]

Step 4: Add Channel Dimension
X_images = X_images[..., np.newaxis]

CNN expects shape: (height, width, channels)
Before: (9, 9)       # Just height × width
After:  (9, 9, 1)    # Height × width × 1 channel (grayscale)

Final Result:
X_images.shape = (225745, 9, 9, 1)
# 225,745 images, each 9×9 pixels, 1 color channel
```


```python
# 6. Save Grayscale Flow Images
```


```python
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
```

    [INFO] Saving grayscale flow images...
    


```python
Explanation:
1. Save First 300 Images: Loop through first 300 samples for visualization
2. Label Determination: Name file based on class (ddos or benign)
3. Convert to 8-bit Image:
img = np.uint8(255 * X_images[i].squeeze())

- Values are currently [0, 1] (float)
- Multiply by 255 to get [0, 255] range
- Convert to uint8 (unsigned 8-bit integer)
- .squeeze() removes the channel dimension for saving

4. Upscale for Viewing:
cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

- Resize from 9×9 to 224×224
- INTER_NEAREST: Keeps blocky/pixelated look (shows structure)

5. Save to Disk: Create PNG files in output/grayscale_images/
Output Files:
output/grayscale_images/
├── benign_0.png
├── benign_1.png
├── ddos_100.png
├── ddos_101.png
└── ...

Visual Examples of Generated Images:

1. Benign Traffic Image
- Darker regions: Lower feature values
- Lighter regions: Higher feature values
- Pattern: Usually more uniform, less extreme values

2. DDoS Attack Image
- High contrast: Extreme values (very dark and very light areas)
- Pattern: Often shows repetitive structures
- Specific features spike: Certain pixels consistently bright

The CNN learns to recognize these visual patterns.
```


```python
# 7. Train / Test Split
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X_images,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```


```python
Explanation:
1. test_size=0.2: 20% of data for testing, 80% for training
2. random_state=42: Seed for reproducibility (always same split)
3. stratify=y: Maintain class balance in both sets

Example Split:
Total samples: 225,745
Training: 180,596 (80%)
Testing:   45,149 (20%)

Both sets have same ratio of DDoS:Benign
```


```python
# 8. Build CNN Model
```


```python
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
```

    [INFO] Building CNN model...
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "CNN_DDoS_Detector"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)             │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">320</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)            │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5184</span>)                │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">663,680</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">129</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">683,009</span> (2.61 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">682,817</span> (2.60 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">192</span> (768.00 B)
</pre>




```python
Explanation:
1. Input Layer
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

- Accepts 9×9×1 grayscale images
- Shape: (None, 9, 9, 1) where None = batch size

2. First Convolutional Layer
x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1")(inputs)

Parameters:
- 32: Number of filters (learns 32 different patterns)
- (3, 3): Filter size (3×3 pixel window)
- activation="relu": ReLU activation (outputs max(0, x))
- padding="same": Keep output size same as input

What it does:
- Slides 3×3 filter across the image
- Each filter detects different patterns (edges, textures)
- Output shape: (None, 9, 9, 32) - 32 feature maps

3. Batch Normalization
x = BatchNormalization()(x)

Purpose:
- Normalizes activations between layers
- Faster training, better stability
- Reduces internal covariate shift

4. Second Convolutional Layer
x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2")(x)
x = BatchNormalization()(x)

What's different:
- 64 filters instead of 32 (learns more complex patterns)
- Operates on 32 input channels (from previous layer)
- Output shape: (None, 9, 9, 64) - 64 feature maps

Pattern hierarchy:
- Conv1 learns: Basic patterns (edges, spots)
- Conv2 learns: Complex patterns (combinations of basic patterns)

5. Flatten Layer
x = Flatten()(x)

Transformation:
Input:  (None, 9, 9, 64)
Output: (None, 5184)    # 9 × 9 × 64 = 5,184 values

Converts 3D tensor to 1D vector for dense layers

6. Dense Hidden Layer
x = Dense(128, activation="relu")(x)

Fully connected layer:
- 128 neurons
- Each neuron connects to all 5,184 inputs
- Learns high-level feature combinations
- ReLU activation for non-linearity

7. Dropout Layer
x = Dropout(0.5)(x)

Regularization technique:
- Randomly drops 50% of neurons during training
- Prevents overfitting
- Forces network to learn robust features

8. Output Layer
outputs = Dense(1, activation="sigmoid")(x)

Binary classification:
- 1 neuron (binary output)
- Sigmoid activation: outputs value between 0 and 1
- Interpretation: Probability of DDoS attack

Output < 0.5 → Benign (class 0)
Output ≥ 0.5 → DDoS (class 1)

Model Compilation:
Components:
1. Optimizer: Adam
- Adaptive learning rate
- Combines momentum and RMSprop
- Efficient and widely used
2. Loss: Binary Cross-Entropy
- Standard for binary classification
- Formula: -[y*log(ŷ) + (1-y)*log(1-ŷ)]
- Penalizes wrong predictions
3. Metrics: Accuracy
- Percentage of correct predictions
- Simple evaluation metric
```


```python
# 9. Train CNN Model
```


```python
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
```

    [INFO] Training CNN...
    Epoch 1/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 82ms/step - accuracy: 0.9849 - loss: 0.0414 - val_accuracy: 0.9893 - val_loss: 0.2517
    Epoch 2/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 82ms/step - accuracy: 0.9912 - loss: 0.0205 - val_accuracy: 0.9898 - val_loss: 0.0368
    Epoch 3/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 83ms/step - accuracy: 0.9943 - loss: 0.0140 - val_accuracy: 0.9956 - val_loss: 0.0086
    Epoch 4/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 84ms/step - accuracy: 0.9979 - loss: 0.0066 - val_accuracy: 0.9997 - val_loss: 0.0023
    Epoch 5/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 89ms/step - accuracy: 0.9988 - loss: 0.0041 - val_accuracy: 0.9996 - val_loss: 0.0031
    Epoch 6/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 82ms/step - accuracy: 0.9993 - loss: 0.0025 - val_accuracy: 0.9998 - val_loss: 0.0011
    Epoch 7/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 87ms/step - accuracy: 0.9996 - loss: 0.0018 - val_accuracy: 0.9998 - val_loss: 4.6051e-04
    Epoch 8/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 83ms/step - accuracy: 0.9995 - loss: 0.0019 - val_accuracy: 0.9998 - val_loss: 9.2596e-04
    Epoch 9/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 81ms/step - accuracy: 0.9996 - loss: 0.0016 - val_accuracy: 0.9997 - val_loss: 0.0013
    Epoch 10/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 90ms/step - accuracy: 0.9997 - loss: 0.0016 - val_accuracy: 0.9998 - val_loss: 4.7297e-04
    Epoch 11/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 102ms/step - accuracy: 0.9996 - loss: 0.0013 - val_accuracy: 0.9996 - val_loss: 0.0016
    Epoch 12/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 102ms/step - accuracy: 0.9996 - loss: 0.0016 - val_accuracy: 0.9998 - val_loss: 7.0261e-04
    Epoch 13/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 99ms/step - accuracy: 0.9998 - loss: 0.0013 - val_accuracy: 0.9998 - val_loss: 5.0495e-04
    Epoch 14/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 101ms/step - accuracy: 0.9998 - loss: 0.0012 - val_accuracy: 0.9994 - val_loss: 0.0019
    Epoch 15/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 92ms/step - accuracy: 0.9995 - loss: 0.0020 - val_accuracy: 0.9998 - val_loss: 6.3907e-04
    Epoch 16/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 83ms/step - accuracy: 0.9997 - loss: 0.0011 - val_accuracy: 0.9996 - val_loss: 0.0020
    Epoch 17/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 87ms/step - accuracy: 0.9998 - loss: 0.0013 - val_accuracy: 0.9998 - val_loss: 7.2086e-04
    Epoch 18/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 82ms/step - accuracy: 0.9998 - loss: 0.0010 - val_accuracy: 0.9953 - val_loss: 0.0121
    Epoch 19/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 80ms/step - accuracy: 0.9996 - loss: 0.0015 - val_accuracy: 0.9998 - val_loss: 5.4573e-04
    Epoch 20/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 84ms/step - accuracy: 0.9998 - loss: 0.0010 - val_accuracy: 0.9997 - val_loss: 0.0013
    Epoch 21/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 82ms/step - accuracy: 0.9998 - loss: 0.0013 - val_accuracy: 0.9999 - val_loss: 0.0012
    Epoch 22/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 80ms/step - accuracy: 0.9998 - loss: 0.0011 - val_accuracy: 0.9998 - val_loss: 4.5509e-04
    Epoch 23/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 82ms/step - accuracy: 0.9998 - loss: 0.0010 - val_accuracy: 0.9997 - val_loss: 0.0015
    Epoch 24/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 84ms/step - accuracy: 0.9998 - loss: 9.0293e-04 - val_accuracy: 0.9998 - val_loss: 5.8260e-04
    Epoch 25/25
    [1m283/283[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 82ms/step - accuracy: 0.9998 - loss: 8.0445e-04 - val_accuracy: 0.9998 - val_loss: 3.6401e-04
    [INFO] Model saved as cnn_ddos_detector.keras
    


```python
Explanation:
Training Parameters:
1. epochs=25: Complete 25 passes through training data
2. batch_size=512: Process 512 images at once
- Larger batch = faster but needs more memory
- 512 is a good balance
3. validation_split=0.2: Use 20% of training data for validation
- Training data: 64% of total
- Validation data: 16% of total
- Test data: 20% of total (from earlier split)

What to Expect:
- Training time: ~5-10 minutes (depending on hardware)
- Accuracy improves each epoch
- Loss decreases each epoch

Save Model:
Saves complete model to disk:
- Architecture (layer structure)
- Weights (learned parameters)
- Optimizer state

Can be loaded later
```


```python
# 10. Evaluate Model
```


```python
print("[INFO] Evaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    [INFO] Evaluating model...
    [1m1411/1411[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 2ms/step
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     19544
               1       1.00      1.00      1.00     25605
    
        accuracy                           1.00     45149
       macro avg       1.00      1.00      1.00     45149
    weighted avg       1.00      1.00      1.00     45149
    
    


```python
Explanation:
1. Metrics Explained:
- Precision: Of predicted DDoS, how many are actually DDoS?
- Recall: Of actual DDoS, how many did we detect?
- F1-Score: Harmonic mean of precision and recall
- Support: Number of samples in each class

2. Typical Performance:
- Accuracy: 99%+
- Very high precision and recall for both classes
```


```python
# 11. GRAD-CAM Function
```


```python
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
```


```python
Explanation:
Grad-CAM (Gradient-weighted Class Activation Mapping) shows which parts of the image the CNN focused on when making
predictions.

How Grad-CAM Works
1. Create Gradient Model:
- Output both convolution layer activations AND final prediction
- This allows us to trace back importance
2. Compute Gradients:
- Calculate how much each pixel in conv layer affects the output
- Higher gradient = more important for prediction
3. Pool Gradients:
- Average gradients across spatial dimensions
- Get importance weight for each filter
4. Create Heatmap:
- Multiply conv outputs by importance weights
- Sum across all filters
- Normalize to [0, 1] range
5. Interpretation:
- Red/Yellow areas: High importance (CNN focused here)
- Blue/Dark areas: Low importance (CNN ignored these)
```


```python
# 12. Save GRAD-CAM Overlay Images
```


```python
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

    # Resize heatmap
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

    # Overlay heatmap on image
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
```

    [INFO] Generating Grad-CAM images...
    [SUCCESS] Grayscale images, CNN model, and Grad-CAM heatmaps generated.
    


```python
Explanation:
Visualization Steps:
1. Generate Heatmap: Use Grad-CAM function
2. Resize Heatmap: 9×9 - 224×224 for visibility
3. Apply Color Map:
cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
- Converts grayscale heatmap to color
- JET colormap: Blue (low) → Red (high)
4. Prepare Original Image:
- Resize to 224×224
- Convert grayscale to BGR (3 channels)
5. Overlay:
cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
- 60% original image
- 40% heatmap
- Creates transparent overlay effect
6. Save: Create PNG files in output/gradcam_images/
```


```python
# Results and Interpretation
```


```python
Expected Outputs
1. Grayscale Images
output/grayscale_images/
├── benign_0.png    (224×224, shows normalized traffic pattern)
├── benign_1.png
├── ddos_100.png    (224×224, shows attack pattern)
└── ...

What to Look For:
- Benign: More uniform, balanced brightness
- DDoS: High contrast, repetitive patterns, bright clusters

2. Trained Model
cnn_ddos_detector.keras (3-5 MB file)

Can be used for real-time detection:
model = tf.keras.models.load_model("cnn_ddos_detector.keras")
prediction = model.predict(new_image)

3. Grad-CAM Images
output/gradcam_images/
├── gradcam_benign_0.png   (Shows what CNN looked at)
├── gradcam_ddos_5.png
└── ...

Interpretation:
- Red regions: CNN thinks this area indicates DDoS
- Blue regions: Not important for classification
- Helps understand what features matter
```


```python
# Usage Instructions
```


```python
Step 1: Prepare Environment
Install dependencies
Create project directory

Step 2: Download Dataset
Place Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv in project directory
Dataset available from: CICIDS2017

Step 3: Run the Code

Step 4: Monitor Training
Watch terminal output:
[INFO] Loading dataset...
[INFO] Normalizing features...
[INFO] Saving grayscale flow images...
[INFO] Building CNN model...
[INFO] Training CNN...
Epoch 1/25
352/352 [==============================] - 45s 128ms/step - loss: 0.1234 - accuracy: 0.9567
...

Step 5: Review Results
Check generated files:
output/grayscale_images/: View traffic patterns
output/gradcam_images/: Understand CNN decisions
cnn_ddos_detector.keras: Use for deployment
```


```python
# Conclusion
```


```python
This guide demonstrates how to:
1. Convert network traffic CSV data into images
2. Train a CNN to recognize DDoS patterns
3. Visualize what the CNN learned with Grad-CAM
4. Achieve 99%+ accuracy on DDoS detection
```
