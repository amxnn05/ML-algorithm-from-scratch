from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier()

bg_model = BaggingClassifier(dt, n_estimators=10, random_state=42)

bg_model.fit(x_train, y_train)
y_pred = bg_model.predict(x_test)

print("accuracy : ", accuracy_score(y_test, y_pred) * 100,"%")
