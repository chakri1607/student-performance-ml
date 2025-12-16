import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Sleep": [6,6,7,7,8,8,9,9],
    "Marks": [35,40,50,60,65,70,75,80],
    "Pass":  [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Inputs and Output
X = df[["Hours", "Sleep"]]   # features
y = df["Pass"]               # target (0 or 1)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Actual Result:", y_test.values)
print("Predicted Result:", y_pred)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Real-life prediction
print("Prediction for 3 hours study & 6 hours sleep:",
      model.predict([[3, 6]]))   # likely Fail (0)

print("Prediction for 6 hours study & 8 hours sleep:",
      model.predict([[6, 8]]))   # likely Pass (1)

