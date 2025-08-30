import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Scratch_SVM:
    # parameters
    def __init__(self, lr = 0.001, lamda = 0.01, epochs = 1000):
        self.lr = lr
        self.lamda = lamda
        self.epochs = epochs
        self.m = None
        self.b = 0

    # training the model
    def fit(self, X, y):
        y_ = np.where(y <= 0 ,-1 ,1)
        self.w = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            for ind, xi in enumerate(X):
                chk = y_[ind] * (np.dot(xi, self.w) + self.b ) >= 1
                if chk:
                    self.w -= self.lr * (2 * self.lamda * self.w)
                else:
                    self.w -= self.lr * (2 * self.lamda * self.w) - ( np.dot(xi, y_[ind]))
                    self.b += self.lr * y_[ind]



    # prediction the model
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


# loading the dataset
data = load_iris()
X = data.data[:, :2]
y = data.target

# converting output to 2 classes -1 and 1
y = np.where(y == 0, -1, 1)

# visualization of the dataset

print("data visualization")
plt.scatter(X[:, 0], X[:, 1], c = y , cmap='bwr')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset')
plt.show()

svm = Scratch_SVM(lr = 0.001, lamda=0.01, epochs=1000)
svm.fit(X, y)
new_samples = np.array([[0, 0], [4, 4]])
svm_pred = svm.predict(new_samples)
print("predicted value using my own class: ", svm_pred)

#using svc class of scikit learn for comparision
sk_svm = SVC()
sk_svm.fit(X, y)
sk_pred = sk_svm.predict(new_samples)
print("predicted value using scikit learn class: ", sk_pred)








## plotting purpose -- copied b/c its boring part :)
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.predict(xy).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.show()

print("after applying my own svm class ")
plot_decision_boundary(X, y, svm)