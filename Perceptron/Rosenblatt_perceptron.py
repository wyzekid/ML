import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, clf, res=0.02):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 2)
        print(np.where(self.net_input(X) >= 0.0, 1, -1))

# setosa and versicolor
star_count = int(input('Enter number of stars: '))
input_matrix = [[0]*3]*star_count
X = []
y = []
for i in range(star_count):
    input_matrix[i] = input().split()
    X.append(input_matrix[i][:2])
    y.append(input_matrix[i][2])

y = np.array(y, dtype=float).reshape(star_count, 1)
X = np.array(X, dtype=float).reshape(star_count, 2)
#map(lambda y: ['-1' if x == 1], y)
print(X)
print(y)
ppn = Perceptron(epochs=10, eta=0.1)

ppn.train(X, y)
plot_decision_regions(X, y, clf=ppn)
plt.title('Perceptron')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()


