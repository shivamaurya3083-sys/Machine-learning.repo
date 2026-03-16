import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Salary": [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Experience"]]
y = df["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Predicted Salary:", predictions)

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Model")
plt.show()