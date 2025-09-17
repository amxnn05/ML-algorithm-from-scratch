import numpy as np
import matplotlib.pyplot as plt
class KNN:
    def __init__(self, k = 5, d = 'ed', p = 2):
        self.k = k
        self.p = p
        self.classes = None
       


# distance calculation using diffrent methods --
    def euclidean_dis(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def manhattan_dis(self, x, y):
        return np.sum(np.abs(x - y), axis = 1)

    def minkowki_dis(self, x, y):
        return np.power(np.sum(np.power(np.abs(x - y), self.p), axis = 1), 1 / self.p)




# predicting the data --
    def proby(self, X, y, predict , k):  
        prob = []
        dis_arr = []
        for i in range(len(X)):
            distance = self.euclidean_dis(np.array(predict),np.array(X[i]))
            dis_arr.append((distance, y[i]))

            
        dis_arr.sort(key = lambda x : x[0])
        n_class = len(np.unique(y))
        kn = []
        for i in range(k):
            kn.append(dis_arr[i][1])

        cls, count = np.unique(kn, return_counts=True)
        for i in range(len(count)): prob.append((cls[i], count[i] / sum(count))) 
        return prob

    def predict(self, X, y, predict, k):
        probs = self.proby(X, y, predict, k)
        for i in range(len(probs)):
            print("probability of being ", probs[i][0] ,"is", probs[i][1] * 100 , "%")
        
## data for training -- 

X = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
y = ['A', 'A', 'A', 'B', 'B']
predict_point = [4, 5]
k = 3
knn = KNN()
knn.predict(X, y, predict_point, 3)

# plotting the data 
# X_A = [pt for pt, label in zip(X, y) if label == 'A']
# X_B = [pt for pt, label in zip(X, y) if label == 'B']
# plt.scatter([p[0] for p in X_A], [p[1] for p in X_A], color='blue', label='Class A')
# plt.scatter([p[0] for p in X_B], [p[1] for p in X_B], color='red', label='Class B')
# plt.scatter(predict_point[0], predict_point[1], color='green', marker='x', s=100, label='Predict Point')
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend()
# plt.grid(True)
# plt.show()