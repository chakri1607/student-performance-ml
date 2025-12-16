import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
y = df["Marks"]              # target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Actual Marks:", y_test.values)
print("Predicted Marks:", y_pred)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Real prediction
print("Prediction for 7 hours study & 8 hours sleep:",
      model.predict([[7, 8]]))
