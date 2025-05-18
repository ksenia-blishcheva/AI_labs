import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# part_1
# 1)
class RosenblattPerceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100,
                 random_weights=True, activation='step', random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_weights = random_weights
        self.activation = activation
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_ = []

    def _initialize_weights(self, n_features):
        if self.random_weights:
            rgen = np.random.RandomState(self.random_state)
            self.weights = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        else:
            self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _activate(self, net_input):
        if self.activation == 'step':

            return np.where(net_input >= 0.0, 1, 0)
        elif self.activation == 'sign':

            return np.where(net_input >= 0.0, 1, -1)
        elif self.activation == 'sigmoid':

            return 1.0 / (1.0 + np.exp(-net_input))
        else:
            raise ValueError(f"unknown activation function: {self.activation}")

    def _net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        # initializing weights
        self._initialize_weights(X.shape[1])

        for epoch in range(self.n_epochs):
            epoch_errors = 0
            for xi, target in zip(X, y):
                # calculating the prediction
                net_input = self._net_input(xi)
                output = self._activate(net_input)

                # updating the weights according to Rosenblatt's rule
                update = self.learning_rate * (target - output)
                self.weights += update * xi
                self.bias += update

                # error counting
                epoch_errors += int(update != 0.0)

            self.errors_.append(epoch_errors)
            if epoch_errors == 0:
                break  # if there are no errors

        return self

    def predict(self, X):
        net_input = self._net_input(X)
        return self._activate(net_input)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 2)
n_samples = 500
data, labels = make_blobs(n_samples=n_samples,
                          centers=([1.1, 3], [4.5, 6.9]),
                          cluster_std=1.3,
                          random_state=0)

colours = ('green', 'orange')
fig, ax = plt.subplots(figsize=(10, 6))

for n_class in range(2):
    ax.scatter(data[labels == n_class][:, 0],
               data[labels == n_class][:, 1],
               c=colours[n_class],
               s=50,
               label=str(n_class))

ax.set_title('raw data')
ax.legend()
plt.show()

# Data separation

# Creating and training a perceptron
perceptron = RosenblattPerceptron(learning_rate=0.1,
                                  n_epochs=100,
                                  random_weights=True,
                                  activation='step')
perceptron.fit(data, labels)

# Prediction and estimation of accuracy
accuracy = perceptron.score(data, labels)
print(accuracy)

# part_2
def plot_decision_boundary(X, y, classifier, resolution=0.02):
    markers = ('s', 'x')
    colors = ('orange', 'green')
    cmap = plt.cm.colors.ListedColormap(colors)

    # graph boundaries
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # building a selection
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

plt.figure(figsize=(10, 6))
plot_decision_boundary(data, labels, classifier=perceptron)
plt.title('the boundary of the perceptron solution')
plt.xlabel('sign 2')
plt.ylabel('sign 2')
plt.legend()
plt.show()

# using a perceptron from Scikit-Learn
sk_perceptron = SkPerceptron(max_iter=100, random_state=0)
sk_perceptron.fit(data, labels)
sk_predictions = sk_perceptron.predict(data)
sk_accuracy = accuracy_score(labels, sk_predictions)
print(f"accuracy_Scikit-Learn: {sk_accuracy:.2f}")
print(f"difference in accuracy: {abs(accuracy - sk_accuracy):.2f}")

# data set Iris
iris = load_iris()
X_iris = iris.data[-100:]
y_iris = iris.target[-100:]
iris_perceptron = SkPerceptron(max_iter=100, random_state=0)
iris_perceptron.fit(X_iris, y_iris)
iris_perceptron_pred = iris_perceptron.predict(X_iris)
iris_perceptron_acc = accuracy_score(y_iris, iris_perceptron_pred)
svm = SVC(kernel='linear')
svm.fit(X_iris, y_iris)
svm_pred = svm.predict(X_iris)
svm_acc = accuracy_score(y_iris, svm_pred)

print(f"accuracy_perceptron: {iris_perceptron_acc:.2f}")
print(f"accuracy_SVM: {svm_acc:.2f}")
print(f"difference in accuracy: {abs(iris_perceptron_acc - svm_acc):.2f}")