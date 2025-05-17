from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Дополнение к предыдущему коду (после оценки качества моделей)
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    w1 = (((1/n * sum([xi * sum_y for xi in x])) - sum_xy) /
          ((1/n * sum([xi * sum_x for xi in x])) - sum_x_squared))
    w0 = (sum_y - w1 * sum_x) / n

    return float(w0), float(w1)

# 1. Data
data = datasets.load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f'INFO: {df.describe()}')

# data selection
print(f'Столбцы: {df.columns}')
# choose 'bmi' и 'target'
x = df[['bmi']].values
y = df['target'].values

# 2.
# implementation using Scikit-Learn
# dividing the data into training and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=45)

# creating and training a model
sklearn_model = LinearRegression()
sklearn_model.fit(x_train, y_train)

# getting coefficients
sk_w0 = float(sklearn_model.intercept_)
sk_w1 = float(sklearn_model.coef_[0])

print(f'coefficients are 1 way: {sk_w0, sk_w1}')

# implementation in its own way
w0, w1 = linear_regression(x, y)
print(f'coefficients are 2 way: {w0, w1}')

# 1. Calculating metrics for the Scikit-Learn model
y_pred_sklearn = sklearn_model.predict(x_test)
sklearn_mae = mean_absolute_error(y_test, y_pred_sklearn)
sklearn_r2 = r2_score(y_test, y_pred_sklearn)
sklearn_mape = np.mean(np.abs((y_test - y_pred_sklearn) / y_test)) * 100

# 2.Calculating metrics for your own model
y_pred_custom = w0 + w1 * x_test.ravel()
custom_mae = mean_absolute_error(y_test, y_pred_custom)
custom_r2 = r2_score(y_test, y_pred_custom)
custom_mape = np.mean(np.abs((y_test - y_pred_custom) / y_test)) * 100

# 3. result
print(f"{'Metrics':<10} {'Scikit-Learn':<15} {'own model':<15}")
print(f"{'MAE':<10} {sklearn_mae:<15.4f} {custom_mae:<15.4f}")
print(f"{'R2':<10} {sklearn_r2:<15.4f} {custom_r2:<15.4f}")
print(f"{'MAPE (%)':<10} {sklearn_mape:<15.2f} {custom_mape:<15.2f}")