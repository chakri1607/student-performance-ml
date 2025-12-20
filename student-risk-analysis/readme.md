# Student Performance Risk Analysis (Machine Learning)

This project focuses on identifying students who are at risk of failing by analyzing
study habits and academic behavior. I built this project to practice applying Machine
Learning end-to-end, from data analysis to model selection and explanation.

## What the Project Does
- Analyzes student behavior data (study hours, sleep, attendance)
- Predicts pass/fail outcome using multiple ML models
- Compares models to select the most reliable one
- Explains model decisions using feature importance
- Visualizes patterns to support insights

## Dataset
The dataset includes:
- Hours: Study hours per day
- Sleep: Hours of sleep
- Attendance: Class attendance percentage
- Marks: Exam score
- Pass: Pass (1) or Fail (0)

The data is created to clearly understand model behavior and risk patterns.

## Models Used
- Logistic Regression (baseline, scaled features)
- Decision Tree (rule-based, prone to overfitting)
- Random Forest (final selected model due to stability)

## Why Random Forest Was Selected
Although Decision Trees achieved high accuracy, they are prone to overfitting.
Random Forest provided stable performance by combining multiple trees and
was chosen as the final model.

## Key Insights
- Attendance and study hours are the strongest predictors of student success
- Sleep contributes but has lower impact compared to attendance
- Model comparison is critical for trustworthy predictions

## Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Git & GitHub

## How to Run
```bash
python3 data.py
python3 eda.py
python3 preprocessing.py
python3 models.py
python3 comparison.py
python3 feature_importance.py
python3 visualization.py

