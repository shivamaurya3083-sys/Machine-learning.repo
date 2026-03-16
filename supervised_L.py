import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Score": [50,55,65,70,75,78,85,90]
}

df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[["Hours"]]
y = df["Score"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print("Predictions:", predictions)

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Supervised Learning Example")
plt.show()