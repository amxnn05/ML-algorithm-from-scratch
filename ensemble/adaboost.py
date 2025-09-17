from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_classification

class ada_boost():
        def __init__(self, n_estimator = 50):
                self.n_estimator = n_estimator
                self.alpha = []
                self.model = []
        
        def fit(self, X ,y):
                n_samples , n_features = X.shape
                w = np.ones(n_samples) / n_samples
                for _ in range(self.n_estimator):
                    model = DecisionTreeClassifier(max_depth = 1)
                    model.fit(X, y, sample_weight=w)
                    pred = model.predict(X)

                    err = np.sum(w * (pred != y)) / np.sum(w)
                    alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
                    self.alpha.append(alpha)
                    self.model.append(model)
        
                    w *= np.exp(- alpha * y * pred)
                    w /= np.sum(w)

        def predict(self, X):
            strong_pred = np.zeros(X.shape[0])
               
            for model, alpha in zip(self.model, self.alpha):
                    predict = model.predict(X)
                    strong_pred += alpha * predict

            return np.sign(strong_pred).astype(int)







X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = ada_boost(n_estimator=50)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("accuracy is ",accuracy_score(y_test, y_pred))
print("precition is ",precision_score(y_test, y_pred))

# dc = DecisionTreeClassifier(max_depth=1)

# ada = AdaBoostClassifier(dc, n_estimators=50, learning_rate=1, random_state=42)
# ada.fit(x_train, y_train)
# y_pred = ada.predict(x_test)

# print("accuracy: ", accuracy_score(y_test, y_pred))
