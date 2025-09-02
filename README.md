# 🩺 Multiclass Diabetes Classification

## 📌 Overview
This project focuses on building a machine learning model to classify patients into three categories based on their health features:

- **0 → Non-Diabetic**
- **1 → Pre-Diabetic**
- **2 → Diabetic**

The workflow includes data preprocessing, exploratory data analysis (EDA), feature selection, handling class imbalance, and model training/tuning.  
Multiple models were compared, and **Gradient Boosting** was chosen as the best-performing algorithm.

---

## 📊 Dataset
- **Total Samples**: 264  
- **Features**: 11 (including clinical and demographic information)  
- **Target**: Diabetes status (0, 1, 2)
  
📂 Dataset: The dataset used in this project can be downloaded from Kaggle: [Multiclass Daibetes Dataset](https://www.kaggle.com/datasets/yasserhessein/multiclass-diabetes-dataset/data)

### Key Features
- **HbA1c** → Glycated hemoglobin (%), strong indicator of diabetes  
- **BMI** → Body Mass Index, linked to obesity  
- **Age** → Patient’s age in years  
- **Urea, Creatinine (Cr)** → Kidney function markers  
- **Cholesterol (Chol), LDL, HDL, TG, VLDL** → Lipid profile  
- **Gender**

✅ Dataset was clean: no missing values, no duplicates.

---

## 🔎 Exploratory Data Analysis (EDA)
- **Class Distribution**: Imbalanced → Class 2 (Diabetic) is majority, Class 1 (Pre-Diabetic) is minority.  
- **Outliers**: Detected in Urea, Creatinine, HbA1c, TG, HDL.  
- **Feature Trends**:  
  - HbA1c and BMI showed clear separation across classes.  
  - Age increased progressively from Class 0 → Class 2.  
  - LDL and HDL contributed very little to prediction power.

---

## ⚙️ Preprocessing Steps
1. **Feature Selection**  
   - Kept: `['HbA1c', 'BMI', 'Age', 'Urea', 'Chol', 'VLDL', 'TG', 'Cr', 'LDL']`  
   - Dropped: LDL, HDL (low importance)

2. **Handling Class Imbalance**  
   - Applied **SMOTE** to oversample minority class (Pre-Diabetic)

3. **Scaling**  
   - StandardScaler applied for Logistic Regression  
   - Not required for tree-based models

---

## 🤖 Models Trained
- Logistic Regression  
- Random Forest  
- XGBoost  
- Gradient Boosting  

Each model was trained **before and after hyperparameter tuning** (RandomizedSearchCV / GridSearchCV).

---

## 📈 Model Comparison

| Model              | Before Tuning Accuracy | After Tuning Accuracy | Before Tuning F1_macro | After Tuning F1_macro |
|-------------------|----------------------|---------------------|----------------------|---------------------|
| Random Forest      | 0.9811               | 0.9811              | 0.98                 | 0.9837              |
| Logistic Regression| 0.9245               | 0.9434              | 0.89                 | 0.92                |
| XGBoost            | 0.9623               | 0.9623              | 0.96                 | 0.96                |
| **Gradient Boosting** | **0.9623**        | **0.9811**          | **0.96**             | **0.9902**          |

---

## 🏆 Why Gradient Boosting is the Best Model
Although Random Forest also achieved **98.1% accuracy**, **Gradient Boosting** was chosen as the final model because:

- **Better F1_macro (0.9902)** → Handles class imbalance effectively  
- **Boosting vs Bagging** → Builds trees sequentially, correcting errors at each step  
- **Generalization** → Slightly better at preventing overfitting compared to Random Forest  
- **Medical Relevance** → Higher recall for minority class (Pre-Diabetic) ensures fewer patients are misclassified  

---

## 📌 Key Insights
1. **HbA1c, BMI, Age** are the top predictors of diabetes status  
2. **Class imbalance** must be addressed; otherwise, minority (Pre-Diabetic) cases are misclassified  
3. **Gradient Boosting** provides the most reliable performance  
4. Logistic Regression is interpretable but less accurate than tree-based methods  

---

## 🚀 Future Work
- Deploy model using **FastAPI / Streamlit** for real-time predictions  
- Integrate with **electronic health records (EHR)**  
- Try **ensemble stacking** (e.g., combining Gradient Boosting + Logistic Regression)  
- Collect **larger dataset** to improve generalizability  

---

## ⚡ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)  
- **Machine Learning**: Random Forest, Logistic Regression, XGBoost, Gradient Boosting  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: FastAPI  

---

## 📢 Conclusion
This project demonstrates how machine learning can be applied in **healthcare diagnostics** to identify diabetic status.  
Through comparative analysis, **Gradient Boosting** emerged as the most effective model, balancing **accuracy, recall, and robustness**.  

By focusing on **key clinical indicators (HbA1c, BMI, Age)** and handling **imbalanced data**, this project provides a solid foundation for real-world diabetes prediction systems.

---

### ⚠️ Disclaimer
This project is created for **educational and research purposes only**.  
The predictions generated by this model are **not intended for actual medical diagnosis or treatment**.  

Users should **not rely solely on this system** for making healthcare decisions.  
Always consult a **qualified healthcare professional** for any medical concerns or advice.  
The author is **not responsible** for any consequences arising from the use of this project.





