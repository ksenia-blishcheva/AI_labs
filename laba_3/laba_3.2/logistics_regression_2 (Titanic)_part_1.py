import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# №1
df = pd.read_csv('Titanic.csv')

df_cleaned = df.dropna()
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

df_cleaned = df_cleaned.drop(cols_to_drop, axis=1)
df_cleaned['Sex'] = df_cleaned['Sex'].map({'male': 0, 'female': 1})
df_cleaned['Embarked'] = df_cleaned['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

initial_rows = df.shape[0]
final_rows = df_cleaned.shape[0]
data_loss_percent = ((initial_rows - final_rows) / initial_rows) * 100
print(f"Dta loss: {data_loss_percent:.2f}%")

# №2
X = df_cleaned.drop('Survived', axis=1)
y = df_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

X_no_embarked = X.drop('Embarked', axis=1)
X_train_ne, X_test_ne, y_train_ne, y_test_ne = train_test_split(X_no_embarked, y, test_size=0.2, random_state=42)

clf_ne = LogisticRegression(max_iter=1000)
clf_ne.fit(X_train_ne, y_train_ne)

y_pred_ne = clf_ne.predict(X_test_ne)
accuracy_ne = accuracy_score(y_test_ne, y_pred_ne)
print(f"Model accuracy without Embarked: {accuracy_ne:.2f}")