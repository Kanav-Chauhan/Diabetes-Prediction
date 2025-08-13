import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction with SVM")
st.markdown("This app predicts whether a patient has diabetes based on medical input features.")

# ========================= LOAD DATA & MODEL =========================
df = pd.read_csv("diabetes.csv")

with open("diabetes_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# ========================= PRECOMPUTED RESULTS =========================
model_results = {
    "Logistic Regression": 0.7709,
    "SVM (RBF)": 0.7709,
    "Random Forest": 0.7670,
    "Gradient Boosting": 0.7566,
    "KNN": 0.7331
}

kernel_results = {
    "linear": 0.7735,
    "rbf": 0.7709,
    "poly": 0.7487,
    "sigmoid": 0.7071
}

best_kernel = "linear"
best_kernel_acc = kernel_results[best_kernel]

# ========================= USER INPUT =========================
st.subheader("🧮 Predict Diabetes from Input")
st.write("Enter patient details or generate a random sample:")

# Function to generate realistic values
def generate_sample(diabetic=True):
    sample = df[df['Outcome'] == (1 if diabetic else 0)].sample(1).iloc[0]
    return {
        "Pregnancies": float(sample['Pregnancies']),
        "Glucose": float(sample['Glucose']),
        "BloodPressure": float(sample['BloodPressure']),
        "SkinThickness": float(sample['SkinThickness']),
        "Insulin": float(sample['Insulin']),
        "BMI": float(sample['BMI']),
        "DiabetesPedigreeFunction": float(sample['DiabetesPedigreeFunction']),
        "Age": float(sample['Age'])
    }

# Store session state for toggle
if "sample_values" not in st.session_state:
    st.session_state.sample_values = generate_sample(diabetic=True)
    st.session_state.generate_diabetic = False  # Next sample will be non-diabetic

# Random generation button
if st.button("🎲 Generate Random Sample"):
    st.session_state.sample_values = generate_sample(diabetic=st.session_state.generate_diabetic)
    st.session_state.generate_diabetic = not st.session_state.generate_diabetic

# Input layout in 4 columns per row
cols1 = st.columns(4)
pregnancies = cols1[0].number_input("Pregnancies", float(0), float(20), float(st.session_state.sample_values['Pregnancies']), step=1.0)
glucose = cols1[1].number_input("Glucose", float(0), float(200), float(st.session_state.sample_values['Glucose']), step=1.0)
bp = cols1[2].number_input("Blood Pressure", float(0), float(150), float(st.session_state.sample_values['BloodPressure']), step=1.0)
skin = cols1[3].number_input("Skin Thickness", float(0), float(100), float(st.session_state.sample_values['SkinThickness']), step=1.0)

cols2 = st.columns(4)
insulin = cols2[0].number_input("Insulin", float(0), float(900), float(st.session_state.sample_values['Insulin']), step=1.0)
bmi = cols2[1].number_input("BMI", float(0), float(70), float(st.session_state.sample_values['BMI']), step=0.1)
dpf = cols2[2].number_input("Diabetes Pedigree Function", float(0), float(3), float(st.session_state.sample_values['DiabetesPedigreeFunction']), step=0.01)
age = cols2[3].number_input("Age", float(1), float(120), float(st.session_state.sample_values['Age']), step=1.0)

# ========================= PREDICTION =========================
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = best_model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Chance of Diabetes")
    else:
        st.success("✅ Low Chance of Diabetes.")

# ========================= WHY SVM =========================
st.subheader("📌 Why Support Vector Machine (SVM)?")
st.write("""
SVM is chosen because:
- Works well for small to medium datasets.
- Handles both linear and non-linear decision boundaries.
- Less prone to overfitting compared to many algorithms.
- Performs well when the number of features is high.
""")

# ========================= WHY LINEAR KERNEL =========================
st.subheader("📌 Why Linear Kernel?")
st.write(f"""
We compared different SVM kernels using cross-validation:

- **Linear Kernel** gave the best accuracy: **{best_kernel_acc:.4f}**
- RBF kernel was close but slightly lower.
- Poly and Sigmoid performed worse.

Linear kernel is simpler, faster, and effective for datasets where classes are mostly linearly separable.
""")

# ========================= VISUALS =========================
st.subheader("📊 Model Comparison")
fig1, ax1 = plt.subplots()
sns.barplot(x=list(model_results.keys()), y=list(model_results.values()), ax=ax1, palette="viridis")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.70, 0.78)
ax1.set_title("Different Model Accuracies")
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("📊 SVM Kernel Comparison")
fig2, ax2 = plt.subplots()
sns.barplot(x=list(kernel_results.keys()), y=list(kernel_results.values()), ax=ax2, palette="coolwarm")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0.70, 0.78)
ax2.set_title("SVM Kernels Performance")
plt.xticks(rotation=45)
st.pyplot(fig2)

# ========================= APPLICATION =========================
st.subheader("🌍 Real-World Applications")
st.write("""
- **Early detection** of diabetes risk for preventive healthcare.
- **Remote health monitoring** via telemedicine.
- **Integration in wearables** to provide instant risk scores.
- **Resource allocation** for healthcare systems.
""")
