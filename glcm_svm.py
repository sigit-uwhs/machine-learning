import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ===============================
# Konfigurasi
# ===============================
IMG_SIZE = (128, 128)
DISTANCES = [1, 2]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
PROPS = ['contrast', 'correlation', 'energy', 'homogeneity']

# ===============================
# Fungsi Ekstraksi GLCM
# ===============================
def extract_glcm_features(image):
    image = cv2.resize(image, IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        image,
        distances=DISTANCES,
        angles=ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in PROPS:
        features.append(graycoprops(glcm, prop).mean())

    return np.array(features)

# ===============================
# Load Dataset
# ===============================
def load_dataset(base_path):
    X, y = [], []
    class_map = {'diabetes': 1, 'nondiabetes': 0}

    for label in class_map:
        folder = os.path.join(base_path, label)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                features = extract_glcm_features(img)
                X.append(features)
                y.append(class_map[label])

    return np.array(X), np.array(y)

# ===============================
# Load Data
# ===============================
X_train, y_train = load_dataset("/content/drive/MyDrive/dataset/dataset_non_augmentasi/train")
X_valid, y_valid = load_dataset("/content/drive/MyDrive/dataset/dataset_non_augmentasi/valid")
X_test, y_test   = load_dataset("/content/drive/MyDrive/dataset/dataset_non_augmentasi/test")

# ===============================
# Model: GLCM + SVM
# ===============================
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale'))
])

# ===============================
# Training
# ===============================
model.fit(X_train, y_train)

# ===============================
# Evaluasi
# ===============================
y_val_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)

val_acc = accuracy_score(y_valid, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)
