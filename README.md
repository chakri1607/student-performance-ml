# Student Performance Prediction – Machine Learning Project

This project was built to understand and apply the complete Machine Learning workflow,
from data analysis to model building and evaluation.

The goal of the project is to predict:
- Student exam marks (Regression)
- Whether a student will pass or fail (Classification)

based on simple features like study hours and sleep hours.

---

## Why I Built This Project
I created this project to strengthen my Machine Learning fundamentals by working on a
realistic problem. Instead of only learning theory, I wanted to practice how data is
analyzed, models are trained, and results are evaluated in a real ML pipeline.

---

## What This Project Does
- Analyzes student performance data
- Predicts exam marks using Linear Regression
- Predicts pass/fail outcome using Logistic Regression
- Evaluates model performance using standard ML metrics
- Visualizes data trends and model behavior

---

## Dataset Description
The dataset includes the following columns:
- **Hours** – Number of hours studied
- **Sleep** – Number of hours slept
- **Marks** – Exam score
- **Pass** – Pass (1) or Fail (0)

The data is created manually to clearly understand the behavior of ML models.

---

## Machine Learning Models Used

### Linear Regression
- Used to predict student marks
- Evaluated using Mean Squared Error (MSE) and R² score

### Logistic Regression
- Used to classify whether a student passes or fails
- Evaluated using accuracy and confusion matrix

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Git & GitHub

---

## How to Run the Project
Run the files in the following order:

```bash
python3 data.py
python3 eda.py
python3 regression.py
python3 classification.py
python3 visualization.py

