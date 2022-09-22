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

# Draw correlation matrix to check for multicollinearity
X = data.drop('salary', axis=1)
# From the matrix, we remove all elements with coef == 1.0 (it's the same column) or coef < 0.2 (too low)
# The rest are the variables which should be treated with caution
X.corr().applymap(lambda x: np.nan if x < 0.2 or x == 1.0 else x)

# Target variable
y = data['salary']

# From the correlation matrix above we see that age is highly correlated with rating and experience
# Let's remove these columns or pairs of columns and see what happens
mape_scores = []
for cols in ['age', 'rating', 'experience',
             ['age', 'rating'], ['age', 'experience'], ['rating', 'experience']]:
    # Remove columns
    X_reduced = X.drop(cols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=100)
    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Calculate predicted salary and find out MAPE
    y_test_pred = model.predict(X_test)
    mape_score = mape(y_test, y_test_pred)
    mape_scores.append(mape_score)
    if verbose:
        print(f'Remove: {cols}\n',
            f'MAPE = {mape_score}')

# Print the final result
if __name__ == '__main__':
    print(f'{min(mape_scores)}')
