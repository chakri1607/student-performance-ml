import pandas as pd
import matplotlib.pyplot as plt
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

# ---- Plot 1: Attendance vs Pass ----
plt.scatter(df["Attendance"], df["Pass"])
plt.xlabel("Attendance (%)")
plt.ylabel("Pass (0 = Fail, 1 = Pass)")
plt.title("Attendance vs Pass Outcome")
plt.show()

# ---- Feature Importance Bar Chart ----
X = df[["Hours", "Sleep", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
plt.bar(X.columns, importances)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()

