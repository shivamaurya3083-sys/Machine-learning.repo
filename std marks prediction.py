import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset (Hours studied vs Marks)
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [30,35,40,45,50,55,60,65,70,75]
}

df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[["Hours"]]
y = df["Marks"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict marks
predicted_marks = model.predict(X_test)

print("Predicted Marks:", predicted_marks)

# Predict for new value
new_hours = [[7]]
prediction = model.predict(new_hours)
print("Predicted marks for 7 study hours:", prediction)

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()