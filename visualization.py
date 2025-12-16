import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Sleep": [6,6,7,7,8,8,9,9],
    "Marks": [35,40,50,60,65,70,75,80],
    "Pass":  [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# -------- PLOT 1: Hours vs Marks (Scatter) --------
plt.scatter(df["Hours"], df["Marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours Studied vs Marks")
plt.show()

# -------- PLOT 2: Regression Line (Hours vs Marks) --------
X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

predicted_marks = model.predict(X)

plt.scatter(df["Hours"], df["Marks"])
plt.plot(df["Hours"], predicted_marks)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression: Hours vs Marks")
plt.show()

