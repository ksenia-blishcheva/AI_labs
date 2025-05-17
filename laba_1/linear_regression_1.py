import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# finding the coefficients of a straight line, MSE
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    w1 = (((1/n * sum([xi * sum_y for xi in x])) - sum_xy) /
          ((1/n * sum([xi * sum_x for xi in x])) - sum_x_squared))
    w0 = (sum_y - w1 * sum_x) / n

    return w0, w1

# shape rendering function
def plot_data_and_regression(X, Y, w0, w1):
    plt.figure(figsize=(10, 5))

    # points based on the raw data X, Y
    plt.scatter(X, Y, color='b')

    # straight y = w0 + w1 * x
    y_1 = [w0 + w1 * float(i) for i in X]
    plt.plot(X, y_1, color='c')

    # error squares
    ax = plt.gca()

    for xi, yi, ypi in zip(X, Y, y_1):
        height = yi - ypi
        if height != 0:
            rect = Rectangle((xi, min(yi, ypi)), abs(height), abs(height),
                             linewidth=1, edgecolor='green',
                             facecolor='gray', alpha=0.2)
            ax.add_patch(rect)

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# №1 reading a file
data = pd.read_csv('student_scores.csv', sep=',')

# №2 statistical information about the data used for each column
print(f'INFO: {data.describe()}')

# №3, 4
print(f'Columns: {list(data.columns)}')
x_d = input(f'Select a column for X: ')
y_d = input(f'Select a column for Y: ')

x = [i for i in data[x_d]]
y = [i for i in data[y_d]]

# MSE
# the equation of a straight line y = w0 + w1 * x
# w0 = (sum(yi) - w1 * sum(xi)) / n
# w1 = (n * sum(xi * yi) - sum(xi) * sum(yi)) / (n * sum(xi ** 2) - (sum(xi) ** 2)
w0, w1 = linear_regression(x, y)

# №5, 6
print(plot_data_and_regression(x, y, w0, w1))




