from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# 1)
warnings.filterwarnings('ignore')

warnings.filterwarnings("ignore", category=UserWarning)
data = pd.read_csv('BostonHousing.csv')
X = data.drop('medv', axis=1)
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                   solver='adam', max_iter=2000, random_state=42)
mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MLPRegressor результаты:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 2) configuration
configurations = [
    (10,),          # 1 слой, 10 нейронов
    (50,),          # 1 слой, 50 нейронов
    (100,),         # 1 слой, 100 нейронов
    (50, 50),       # 2 слоя по 50 нейронов
    (100, 50),      # 2 слоя: 100 и 50 нейронов
    (50, 30, 20),   # 3 слоя
    (100, 100, 50)  # 3 слоя
]

results = []
for config in configurations:
    mlp = MLPRegressor(hidden_layer_sizes=config, activation='relu',
                      solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    results.append((config, mse))

#
configs_str = [str(c) for c in configurations]
mses = [r[1] for r in results]

plt.figure(figsize=(12, 6))
plt.bar(configs_str, mses)
plt.xticks(rotation=45)
plt.xlabel('configuration of hidden layers')
plt.ylabel('MSE')
plt.title('the impact of network architecture')
plt.show()

#  3) activation function
activations = ['identity', 'logistic', 'tanh', 'relu']
activation_results = []

for act in activations:
    mlp = MLPRegressor(hidden_layer_sizes=(100,50), activation=act,
                      solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    activation_results.append(mse)

plt.figure(figsize=(10, 5))
plt.bar(activations, activation_results)
plt.xlabel('activation function')
plt.ylabel('MSE')
plt.title('the effect of the activation function')
plt.show()

# 4) optimization algorithm
solvers = ['lbfgs', 'sgd', 'adam']
solver_results = []

for solv in solvers:
    mlp = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu',
                      solver=solv, max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    solver_results.append(mse)

plt.figure(figsize=(10, 5))
plt.bar(solvers, solver_results)
plt.xlabel('optimization algorithm')
plt.ylabel('MSE')
plt.title('the effect of the optimization algorithm')
plt.show()

