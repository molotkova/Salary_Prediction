import os
import requests

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

# Flag to enable printing of diagnostic information
verbose = False

# Fit the best model from Stage 4, replace negative predictions with 0 or with median and see shat happens
# Create X and y
X = data.drop(['salary', 'age', 'experience'], axis=1)
y = data['salary']

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate predicted salary and find out MAPE
y_test_pred = model.predict(X_test)
y_test_pred_0 = np.where(y_test_pred < 0, 0, y_test_pred)
y_test_pred_median = np.where(y_test_pred <0, y_train.median(), y_test_pred)

mape_score = mape(y_test, y_test_pred)
mape_score_0 = mape(y_test, y_test_pred_0)
mape_score_median = mape(y_test, y_test_pred_median)

if verbose:
    print(f'Without replacement: {mape_score}')
    print(f'Replacement with 0: {mape_score_0}')
    print(f'Replacement with median: {mape_score_median}')

# Print the final result
if __name__ == '__main__':
    print(f'{min((mape_score, mape_score_0, mape_score_median))}')