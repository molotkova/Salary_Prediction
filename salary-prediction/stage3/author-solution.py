import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


# Load data
data = pd.read_csv('data.csv')

# Flag to enable printing of diagnostic information
verbose = False

# Now X contains all variables except for salary
X = data.drop('salary', axis=1)
y = data['salary']

# One-hot encoding of categorical variables
X = pd.get_dummies(X.select_dtypes('object')).join(X.select_dtypes('number'))

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate predicted salary and find out MAPE
y_test_pred = model.predict(X_test)
mape_score = mape(y_test, y_test_pred)
if verbose:
      print(f'Model coefficients are {model.coef_}.\n',
            f'MAPE is {mape_score}')

# Print the final result
if __name__ == '__main__':
      print(', '.join(str(coef) for coef in model.coef_))