from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
X = digits.data
y = digits.target
# print(digits.data.shape)
#
plt.gray()
plt.matshow(digits.images[1])
plt.show()
#
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50),  # Два скрытых слоя: 100 и 50 нейронов
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42,
                    verbose=True)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("\noverall accuracy of the model: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('predicted class')
plt.ylabel('the true class')
plt.title('classification error matrix')
plt.show()