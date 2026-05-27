"""
Local training script — run this ONCE to generate digit_model.pkl
which is then committed to the repo and loaded by the Streamlit app.
"""
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

print("Training MLP (this takes ~2-3 min)...")
clf = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=30,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"\nTest accuracy: {score * 100:.2f}%")

joblib.dump(clf, "digit_model.pkl")
print("Saved → digit_model.pkl")
