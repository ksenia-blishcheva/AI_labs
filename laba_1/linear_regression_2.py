import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

# 3. displaying results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue') # raw data

plt.plot(x, sk_w0 + sk_w1 * x, color='red') # regression line Scikit-learn
plt.plot(x, w0 + w1 * x, color='green') # regression line is its own

plt.title('Comparison of linear regression')
plt.xlabel('BMI')
plt.ylabel('Disease progression')
plt.legend()
plt.grid(True)
plt.show()

# 4. A table with the results of predictions
# creating a Data Frame for comparison
predictions = pd.DataFrame({
    'Actual': y_test,
    'Scikit-Learn Prediction': sklearn_model.predict(x_test),
    'Custom Algorithm Prediction': w0 + w1 * x_test.ravel()
})

print(predictions.head())

# 5. Model quality assessment

# For the Scikit-Learn model
sklearn_mse = mean_squared_error(y_test, sklearn_model.predict(x_test))
sklearn_r2 = r2_score(y_test, sklearn_model.predict(x_test))

# For your own model
custom_mse = mean_squared_error(y_test, w0 + w1 * x_test.ravel())
custom_r2 = r2_score(y_test, w0 + w1 * x_test.ravel())

print("\nModel quality assessment:")
print(f"{'Metric':<20} {'Scikit-Learn':<20} {'Own algorithm':<20}")
print(f"{'MSE':<20} {sklearn_mse:<20.4f} {custom_mse:<20.4f}")
print(f"{'R2':<20} {sklearn_r2:<20.4f} {custom_r2:<20.4f}")
