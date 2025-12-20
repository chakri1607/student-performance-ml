import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Sleep": [6, 6, 7, 7, 8, 8, 9, 9],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [35, 40, 50, 60, 65, 70, 75, 80],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Hours", "Sleep", "Attendance"]]
y = df["Pass"]

# Train/Test split FIRST (important rule)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Before scaling (train):")
print(X_train.head())

print("\nAfter scaling (train):")
print(X_train_scaled)

