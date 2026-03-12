import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "Area": [1000, 1500, 1800, 2400, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Age": [10, 5, 8, 2, 1],
    "Price": [200000, 300000, 350000, 450000, 500000]
}

df = pd.DataFrame(data)

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)

print("Predicted Prices:", predictions)
print("Actual Prices:", y_test.values)
print("Mean Absolute Error:", error)

new_house = [[2000, 3, 5]]  # Area, Bedrooms, Age
predicted_price = model.predict(new_house)

print("Predicted price for new house:", predicted_price)