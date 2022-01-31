import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def evaluate(test_y, true_y):
    true_y = true_y.tolist()
    num = len(true_y)
    counter = 0
    for i in range(num):
        if test_y[i] == true_y[i]:
            counter += 1
    print('correct:', counter, 'num:', num)
    return counter / num


class Logistic_Regression():
    def __init__(self):
        self.w = []
        self.epsilon = 0.25
        self.alpha = 0.001
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
        loss = []
        L1 = 0

        for i in range(train_x.shape[0]):
            L1 += np.log2(1 + np.exp((np.dot(train_x[i], w)[0]))) - train_y[i] * (np.dot(train_x[i], w)[0])

        loss.append(L1)

        while True:
            counter += 1
            dl = self.grad(w, train_x, train_y)
            w = w + self.alpha * dl

            L2 = 0
            for i in range(train_x.shape[0]):
                L2 += np.log2(1 + np.exp((np.dot(train_x[i], w)[0]))) - train_y[i] * (np.dot(train_x[i], w)[0])
            loss.append(L2)

            if abs(L2 - L1) < self.epsilon and counter > 50:
                break
            L1 = L2
        self.w = w
        print('训练了', counter, '次')
        times = np.arange(0, counter + 1)
        loss = np.array(loss).flatten()

        plt.title('训练次数与损失函数的关系')
        plt.xlabel('训练次数')
        plt.ylabel('损失函数')
        plt.plot(times, loss)
        plt.show()

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

    def arg(self):
        return self.w


if __name__ == '__main__':
    raw = pd.read_csv('wdbc.data', header=None)
    df1 = raw.sample(frac=0.8)
    df2 = raw[raw.index.isin(df1.index) == False]
    trainx = df1.iloc[:, 2:]
    trainy = df1.iloc[:, 1].apply(lambda x: 1 if x == 'M' else 0)
    testx = df2.iloc[:, 2:]
    testy = df2.iloc[:, 1].apply(lambda x: 1 if x == 'M' else 0)

    LR = Logistic_Regression()
    LR.fit(trainx, trainy)

    pre = LR.predict(testx)
    out = []
    for i in pre:
        if i == 0:
            out.append('B')
        else:
            out.append('M')
    print(out)
    print('correct rate:', evaluate(pre, testy) * 100, '%')
