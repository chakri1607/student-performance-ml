import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Sleep": [6,6,7,7,8,8,9,9],
    "Attendance": [60,65,70,75,80,85,90,95],
    "Marks": [35,40,50,60,65,70,75,80],
    "Pass": [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features & target
X = df[["Hours", "Sleep", "Attendance"]]
y = df["Pass"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("Feature Importance:")
print(feature_importance_df)

