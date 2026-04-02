Machiimport os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# KONFIGURASI
# ===============================
IMG_SIZE = (128, 128)
DATASET_DIR = "/content/drive/MyDrive/dataset/dataset_non_augmentasi"

CLASSES = {
    "diabetes": 1,
    "nondiabetes": 0
}

# LBP parameter
RADIUS = 2
N_POINTS = 8 * RADIUS

# Color histogram
BINS = 16

# ===============================
# EKSTRAKSI FITUR
# ===============================
def extract_features(image):
    # Resize
    image = cv2.resize(image, IMG_SIZE)

    # ---------- HOG ----------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # ---------- LBP ----------
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # ---------- Color Histogram (HSV) ----------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [BINS, BINS, BINS],
        [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(color_hist, color_hist)
    color_hist = color_hist.flatten()

    # ---------- GABUNGKAN ----------
    return np.hstack([hog_feat, lbp_hist, color_hist])

# ===============================
# LOAD DATA
# ===============================
def load_data(split):
    X, y = [], []
    split_path = os.path.join(DATASET_DIR, split)

    for label_name, label_value in CLASSES.items():
        folder = os.path.join(split_path, label_name)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            features = extract_features(img)
            X.append(features)
            y.append(label_value)

    return np.array(X), np.array(y)

# ===============================
# LOAD DATASET
# ===============================
print("Loading training data...")
X_train, y_train = load_data("train")

print("Loading validation data...")
X_valid, y_valid = load_data("valid")

print("Loading test data...")
X_test, y_test = load_data("test")

print("Feature dimension:", X_train.shape[1])

# ===============================
# MODEL: SCALER + SVM
# ===============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])

# ===============================
# TRAINING
# ===============================
print("Training Fusion Feature + SVM...")
model.fit(X_train, y_train)

# ===============================
# VALIDATION
# ===============================
y_valid_pred = model.predict(X_valid)
print("\nValidation Accuracy:", accuracy_score(y_valid, y_valid_pred))

# ===============================
# TESTING
# ===============================
y_test_pred = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=CLASSES.keys()))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
