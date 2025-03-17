import joblib
import os

model_path = "model.pkl"  # Adjust path if needed

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model Loaded Successfully:", type(model))
else:
    print("Model file not found!")
