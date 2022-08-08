import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


# Load data
data = pd.read_csv('data.csv')

# Flag to enable printing of diagnostic information
verbose = False

# Extract predictors and target
X = data[['rating']]
y = data['salary']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fit model #1 (salary ~ rating)
model = LinearRegression()
model.fit(X_train, y_train)
if verbose:
    print(f'The resulting formula is salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} * rating')

# Predict salary on test part and calculate MAPE
y_test_pred = model.predict(X_test)
if verbose:
    print(f'MAPE is {mape(y_test, y_test_pred)}')

# Print the result
if __name__ == '__main__':
    print(f'{model.intercept_:.5f} {model.coef_[0]:.5f} {mape(y_test, y_test_pred):.5f}')