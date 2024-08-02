# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('/Users/gordon/Documents/Studies/DS/Data Glacier/Flask Project/Flask-Deployment2/price.csv')

X = dataset.iloc[:, :3]

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with training data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))