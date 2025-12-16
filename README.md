# Student Performance Prediction using Machine Learning

## 📌 Project Overview
This project predicts student performance using Machine Learning techniques.
It uses study hours and sleep hours to:
- Predict exam marks (Regression)
- Predict pass/fail outcome (Classification)

The project demonstrates a complete end-to-end ML workflow.

---

## 🧠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📊 Dataset
The dataset contains the following features:
- Hours: Number of hours studied
- Sleep: Number of hours slept
- Marks: Exam score
- Pass: Pass (1) or Fail (0)

---

## ⚙️ Machine Learning Models
### 1. Linear Regression
- Used to predict exam marks
- Evaluated using Mean Squared Error (MSE) and R² score

### 2. Logistic Regression
- Used to predict pass/fail outcome
- Evaluated using accuracy and confusion matrix

---

## 📈 Visualization
- Scatter plot of Hours vs Marks
- Regression line to show model predictions

---

## 🚀 How to Run
```bash
python3 data.py
python3 eda.py
python3 regression.py
python3 classification.py
python3 visualization.py

