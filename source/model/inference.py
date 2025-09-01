import joblib
import pandas as pd

model = joblib.load('gb_diabetes_model.joblib')

print("Enter patient details to predict diabetes class:")
print("Please follow the recommended ranges for accurate prediction.")

hba1c = float(input("HbA1c (%) [Normal: 4.5-5.6, Pre-diabetic: 5.7-6.4, Diabetic: >6.5]: "))
bmi = float(input("BMI (kg/m^2) [Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30+]: "))
age = int(input("AGE (years) [Recommended: 20-80]: "))
urea = float(input("Urea (mg/dL) [Normal: 7-20]: "))
chol = float(input("Total Cholesterol (mg/dL) [Normal: <200]: "))
vldl = float(input("VLDL (mg/dL) [Normal: 2-30]: "))
tg = float(input("Triglycerides (mg/dL) [Normal: <150]: "))
cr = float(input("Creatinine (mg/dL) [Normal: 0.6-1.3]: "))
ldl = float(input("LDL (mg/dL) [Normal: <100]: "))

# Create DataFrame for prediction
new_data = pd.DataFrame([{
    'HbA1c': hba1c,
    'BMI': bmi,
    'AGE': age,
    'Urea': urea,
    'Chol': chol,
    'VLDL': vldl,
    'TG': tg,
    'Cr': cr,
    'LDL': ldl
}])

# Predict class
predicted_class = model.predict(new_data)[0]

# Map numeric class to descriptive label
class_mapping = {
    0: "Non-diabetic ✅ Healthy blood sugar levels",
    1: "Pre-diabetic ⚠️ At risk: monitor diet and lifestyle",
    2: "Diabetic ❌ High risk: consult a doctor immediately"
}

print("\nPredicted Class:", class_mapping[predicted_class])