import pandas as pd
import numpy as np
import sys


class Logistic_Regression:
    def __init__(self):
        self.w = []
        self.epsilon = 0.25
        self.alpha = 0.01
        self.max = []
        self.min = []

    def grad(self, w, x, y):
        return ((y - self.sigmoid(x @ w)).T @ x).T

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, train_x, train_y):

        w = np.ones((train_x.shape[1] + 1, 1))
        train_y = np.array(train_y).reshape(len(train_y), 1)

        for i in range(train_x.shape[1]):
            self.max.append(train_x.iloc[:, i].max())
            self.min.append(train_x.iloc[:, i].min())
            train_x.iloc[:, i] = (train_x.iloc[:, i] - train_x.iloc[:, i].min()) / (
                    train_x.iloc[:, i].max() - train_x.iloc[:, i].min())
        train_x = np.array(train_x)
        train_x = np.c_[train_x, np.ones(train_x.shape[0])]

        counter = 0
        L1 = 0

        for i in range(train_x.shape[0]):
            L1 += np.log2(1 + np.exp((np.dot(train_x[i], w)[0]))) - train_y[i] * (np.dot(train_x[i], w)[0])

        while True:
            counter += 1
            dl = self.grad(w, train_x, train_y)
            w = w + self.alpha * dl

            L2 = 0
            for i in range(train_x.shape[0]):
                L2 += np.log2(1 + np.exp((np.dot(train_x[i], w)[0]))) - train_y[i] * (np.dot(train_x[i], w)[0])

            if abs(L2 - L1) < self.epsilon and counter > 50:
                break
            L1 = L2

        self.w = w

    def predict(self, test_x):

        for i in range(test_x.shape[1]):
            test_x.iloc[:, i] = (test_x.iloc[:, i] - self.min[i]) / (self.max[i] - self.min[i])
        test_x = np.array(test_x)
        test_x = np.c_[test_x, np.ones(test_x.shape[0])]

        pre = self.sigmoid(test_x @ self.w).flatten().tolist()
        for i in range(len(pre)):
            if pre[i] > 0.5:
                pre[i] = 1
            else:
                pre[i] = 0
        return pre


if __name__ == '__main__':
    df1 = pd.read_csv(sys.argv[1], header=None)
    df1.iloc[:, 1] = df1.iloc[:, 1].apply(lambda x: 1 if x == 'M' else 0)
    trainx = df1.iloc[:, 2:]
    trainy = df1.iloc[:, 1]

    df2 = pd.read_csv(sys.argv[2], header=None)
    testx = df2.iloc[:, 2:]

    LR = Logistic_Regression()
    LR.fit(trainx, trainy)
    pre = LR.predict(testx)

    for i in pre:
        if i == 0:
            print('B')
        else:
            print('M')
