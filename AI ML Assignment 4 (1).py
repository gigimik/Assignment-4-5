# =====================================================================
# CNN-BASED DDoS DETECTION
# Dataset: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
# Output:
#   - Grayscale flow images
#   - Trained CNN model (.keras)
#   - Grad-CAM heatmap overlay images
# =====================================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
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

# =========================
# 2. CONFIGURATION
# =========================
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

# =========================
# 3. LOAD DATASET
# =========================
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

# =========================
# 4. FEATURE NORMALIZATION
# =========================
print("[INFO] Normalizing features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. CONVERT FLOWS - GRAYSCALE IMAGES
# =========================
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

# =========================
# 6. SAVE GRAYSCALE FLOW IMAGES
# =========================
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

# =========================
# 7. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_images,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 8. BUILD CNN MODEL
# =========================
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

# =========================
# 9. TRAIN CNN MODEL
# =========================
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

# =========================
# 10. EVALUATE MODEL
# =========================
print("[INFO] Evaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 11. GRAD-CAM FUNCTION
# =========================
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

# =========================
# 12. SAVE GRAD-CAM OVERLAY IMAGES
# =========================
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
