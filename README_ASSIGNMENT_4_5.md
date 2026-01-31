# Assignment 4 & Assignment 5
Network Traffic Classification Using Deep Learning

## Course
AI and Machine Learning for Cybersecurity

---

## Assignment 4 – Implementation

### Dataset
For this assignment, the dataset  
`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`  
was utilized.

The dataset contains network traffic features extracted from PCAP files and
includes labeled samples of both normal traffic and DDoS attack traffic.

---

### Preprocessing
- The CSV dataset was loaded using the pandas library
- Non-relevant and non-numeric columns were removed
- Missing and invalid values were processed
- Class labels were encoded
- Numerical features were normalized

---

### CSV to Image Conversion
To apply deep learning methods, the network traffic data was converted
into grayscale image representations:

- Each network flow was reshaped into a **9×9 grayscale image**
- Each image represents the behavior of an individual traffic flow
- This conversion enables the use of a Convolutional Neural Network (CNN)

---

### Model
A **Convolutional Neural Network (CNN)** was trained using TensorFlow/Keras.

The model architecture consists of:
- Convolutional layers
- Batch normalization layers
- Dropout layers
- Fully connected layers

The trained model was saved as:
cnn_ddos_detector.keras


---

### Training and Evaluation
The model was trained on the processed dataset and evaluated using:
- Accuracy
- Classification report
- Confusion matrix

The trained model demonstrated strong performance, achieving approximately
**99% accuracy** in detecting DDoS attacks.

---

## Assignment 5 – Documentation

This section provides documentation for the implementation completed
in Assignment 4.

### Code Workflow
1. The CSV dataset is loaded and preprocessed
2. Network traffic flows are converted into image representations
3. A CNN model is trained on the generated images
4. The trained model is evaluated using standard classification metrics
5. Grad-CAM is used to visualize important regions influencing predictions

---

### How to Run the Project
1. Upload the following files to Google Colab:
   - `AI ML Assignment 4.py`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

2. Run the following command:
   ```bash
   python "AI ML Assignment 4.py"

Libraries Used

Python 3

pandas

numpy

scikit-learn

tensorflow / keras

matplotlib