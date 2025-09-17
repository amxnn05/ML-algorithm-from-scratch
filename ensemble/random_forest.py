import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/titanic.csv')
df = df.dropna(subset = 'Survived')
df.sample(5)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X
X.loc[:,'Sex'] = df['Sex'].map({'female': 0, 'male': 1})
X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))

