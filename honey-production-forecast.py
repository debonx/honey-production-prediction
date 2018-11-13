import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Data about Honey production from 1998 to 2013
df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

# Group by year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year['year']
X = X.values.reshape(-1, 1)
y = prod_per_year['totalprod']

# Scatter data until 2013
plt.scatter(X, y)
plt.show()

# Create Regression with current data
regr = linear_model.LinearRegression()
regr.fit(X, y)
y_predict = regr.predict(X)

# Plot current data
plt.plot(X, y_predict)
plt.show()

# Predict data until 2050
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.xlabel("Year")
plt.ylabel("Pounds of honey production")
plt.show()

# Sad story, we should do something.
