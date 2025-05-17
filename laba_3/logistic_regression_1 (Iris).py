from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

data = datasets.load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
# Добавим столбец "target" и заполним его данными.
df['target'] = data.target
# Выведем на экран первые пять строк.
df.head()

x = data.data
y = data.target
features = data.feature_names
target_names = data.target_names

# №1
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i, name in enumerate(target_names):
    plt.scatter(x[y == i, 0], x[y == i, 1], label=name)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal length vs Sepal width')
plt.legend()

plt.subplot(1, 2, 2)
for i, name in enumerate(target_names):
    plt.scatter(x[y == i, 2], x[y == i, 3], label=name)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal length vs Petal width')
plt.legend()

plt.tight_layout()

# №2
data_df = pd.DataFrame(x, columns=features)
data_df['species'] = [target_names[i] for i in y]

sns.pairplot(data_df, hue='species')
plt.show()

# №3
mask1 = (y == 0) | (y == 1)
x_1, y_1 = x[mask1], y[mask1]
mask2 = (y == 1) | (y == 2)
x_2, y_2 = x[mask2], y[mask2]

# №4-8
def train(x, y, x_train, y_train, x_test, y_test):

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

accuracy_1 = train(x_1, y_1, x_train, y_train, x_test, y_test)
print(accuracy_1)

accuracy_2 = train(x_2, y_2, x_train, y_train, x_test, y_test)
print(accuracy_2)

# №9
x_synth, y_synth = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                      n_informative=2, random_state=1, n_clusters_per_class=1)
# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(x_synth[:, 0], x_synth[:, 1], c=y_synth, cmap='bwr', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

clf = LogisticRegression(random_state=0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

#
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1
)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Dataset')
plt.show()
