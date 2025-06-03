from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
try:
    model = load_model("mask.h5", compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
