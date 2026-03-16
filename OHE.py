import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = [["Red"], ["Blue"], ["Green"], ["Blue"]]

encoder = OneHotEncoder(sparse=False)

encoded = encoder.fit_transform(data)

print(encoded)