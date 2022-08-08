import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


# Load data
data = pd.read_csv('data.csv')

# Flag to enable printing of diagnostic information
verbose = False

# Fit with different degrees of X
mape_scores = []
for degree in range(1, 5):
    # Raise rating to necessary degree
    X = pd.DataFrame(data['rating'] ** degree)
    y = data['salary']

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate predicted salary and find out MAPE
    y_test_pred = model.predict(X_test)
    mape_score = mape(y_test, y_test_pred)
    mape_scores.append((degree, mape_score))

    # Print results
    if verbose:
        print(f'The resulting formula for degree {degree} is',
            f'salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} * rating^{degree}.',
            f'MAPE is {mape_score}')

mape_scores.sort(key=lambda x: x[1])
best_degree, score = mape_scores[0]
if verbose:
    print(f'The model with degree {best_degree} is the best, and its MAPE is {score}')

# Print the final result
if __name__ == '__main__':
    print(f'{score}')