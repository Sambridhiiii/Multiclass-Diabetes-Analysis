import pandas as pd
import joblib
import os
from pathlib import Path

def load_model(model_path: str = "model/gb_diabetes_model.joblib"):
    """Load the trained pipeline from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

class DiabetesPredictor:
    def __init__(self):
        # Load the model
        model_path = os.path.join(os.path.dirname(__file__), "../model/gb_diabetes_model.joblib")
        self.model = load_model(model_path)
        print("Diabetes model loaded successfully!")

    def predict_class(self, user_data):
        df = pd.DataFrame(user_data)
        return self.model.predict(df)

    def map_class(self, predicted_class):
        """Convert numeric prediction to a descriptive label."""
        class_mapping = {
            0: "Non-diabetic ✅ Healthy blood sugar levels",
            1: "Pre-diabetic ⚠️ At risk: monitor diet and lifestyle",
            2: "Diabetic ❌ High risk: consult a doctor immediately"
        }
        return class_mapping.get(predicted_class, "Unknown")

    def get_user_input(self):
        """Ask the user for patient details (beginner-friendly)."""
        user_data = {}
        try:
            print("\nEnter patient details to predict diabetes class:")
            print("Please follow the recommended ranges for accurate prediction.\n")

            user_data['HbA1c'] = float(input("HbA1c (%) [Normal: 4.5-5.6, Pre-diabetic: 5.7-6.4, Diabetic: >6.5]: "))
            user_data['BMI'] = float(input("BMI (kg/m^2) [Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30+]: "))
            user_data['AGE'] = int(input("AGE (years) [Recommended: 20-80]: "))
            user_data['Urea'] = float(input("Urea (mg/dL) [Normal: 7-20]: "))
            user_data['Chol'] = float(input("Total Cholesterol (mg/dL) [Normal: <200]: "))
            user_data['VLDL'] = float(input("VLDL (mg/dL) [Normal: 2-30]: "))
            user_data['TG'] = float(input("Triglycerides (mg/dL) [Normal: <150]: "))
            user_data['Cr'] = float(input("Creatinine (mg/dL) [Normal: 0.6-1.3]: "))
            user_data['LDL'] = float(input("LDL (mg/dL) [Normal: <100]: "))

        except ValueError:
            print("Invalid input! Please enter numbers where required.")
            return None
        return [user_data]

# ------------------ Run Example ------------------
if __name__ == "__main__":
    predictor = DiabetesPredictor()
    user_input = predictor.get_user_input()
    if user_input:
        predicted_class = predictor.predict_class(user_input)[0]
        result = predictor.map_class(predicted_class)

        print("\n--- Prediction Results ---")
        print(f"Predicted Class: {result}")
