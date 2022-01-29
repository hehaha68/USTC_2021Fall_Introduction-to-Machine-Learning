# -*- coding: utf-8 -*-
""" 
@Time    : 2021/10/12 11:44
@Author  : 和泳毅
@FileName: LogisticRegression.py
@SoftWare: PyCharm
"""

import pandas as pd
import numpy as np
import random
import sys


# 改进的sigmoid函数
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


# 逻辑回归
class LogisticRegression:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: 收敛条件
    epochs: 更新迭代的次数
    w: 训练参数
    status_dict: 标签列表
    maxmin: 归一化数据
    '''

    def __init__(self, Lambda=0.001, epochs=1000, lr=0.1):
        self.Lambda = Lambda
        self.epochs = epochs
        self.lr = lr
        self.w = np.ones((31, 1))
        self.status_dict = ['B', 'M']
        self.maxmin = []

    def pretreatment(self, df, flag='train'):
        df_c = df.copy()
        # 添加一列
        df_c['f31'] = 1
        if flag == 'train':
            # 转换标签
            df_c['Label_tran'] = df_c['Label'].apply(lambda x: self.status_dict.index(x))
        # 特征归一化
        for i in range(30):
            if flag == 'train':
                self.maxmin += [[df_c.iloc[:, i + 2].min(axis=0), df_c.iloc[:, i + 2].max(axis=0)]]
            df_c.iloc[:, i + 2] = (df_c.iloc[:, i + 2] - self.maxmin[i][0]) / (self.maxmin[i][1] - self.maxmin[i][0])
        return df_c

    def fit(self, train_features, train_labels):

        def gold_div_search(a, b, esp, N, w, x, y, dleta):
            rou = 1 - (np.sqrt(5) - 1) / 2  # 1-rou为黄金分割比
            lr1 = a + rou * (b - a)
            lr2 = b - rou * (b - a)

            while b - a > esp:

                w1 = w - lr1 * dleta
                w2 = w - lr2 * dleta
                w1_T_x = np.dot(w1.T[0], x)
                w2_T_x = np.dot(w2.T[0], x)
                f1 = 0
                f2 = 0
                for i in range(N):
                    f1 = f1 + (-y[i] * w1_T_x[i] + np.log(1 + np.exp(w1_T_x[i])))
                    f2 = f2 + (-y[i] * w2_T_x[i] + np.log(1 + np.exp(w2_T_x[i])))
                if f1 > f2:  # 如果f(x1)>function(x2)，则在区间(x1,b)内搜索
                    a = lr1
                    lr1 = lr2
                    lr2 = b - rou * (b - a)
                elif f1 < f2:  # 如果f(x1)<function(x2),则在区间(a,x2)内搜索
                    b = lr2
                    lr2 = lr1
                    lr1 = a + rou * (b - a)
                else:  # 如果f(x1)=function(x2)，则在区间(x1,x2)内搜索
                    a = lr1
                    b = lr2
                    lr1 = a + rou * (b - a)
                    lr2 = b - rou * (b - a)
            return a

        x = np.array(train_features).T
        y = np.array(train_labels)
        num, N = np.shape(x)
        w = np.ones((num, 1))

        L = np.zeros(self.epochs + 1)
        L[0] = 1000

        for j in range(self.epochs):
            w_T_x = np.dot(w.T[0], x)

            # 收敛条件
            for i in range(N):
                L[j + 1] = L[j + 1] + (-y[i] * w_T_x[i] + np.log(1 + np.exp(w_T_x[i])))
            if np.abs(L[j + 1] - L[j]) <= self.Lambda:
                break

            # 计算梯度
            dbeta = 0  # 一阶导
            for i in range(N):
                p1 = sigmoid(w_T_x[i])
                dbeta = dbeta - np.array([x[:, i]]).T * (y[i] - p1)
            gk = dbeta

            # 优化参数
            # self.lr = gold_div_search(0, 1, 0.005, N=N, w=w, x=x, y=y, dleta=gk)
            w = w - self.lr * gk

        self.w = w

    def predict(self, test_features):

        x = np.array(test_features).T
        w_T_x = np.dot(self.w.T[0], x)
        pre = np.zeros((x.shape[1], 1))
        y = np.zeros((x.shape[1], 1))

        for i in range(pre.shape[0]):
            pre[i] = sigmoid(w_T_x[i])
            if pre[i] < 0.5:
                y[i] = 0
            elif pre[i] > 0.5:
                y[i] = 1
            else:
                y[i] = random.randint(0, 1)

        y_label = pd.DataFrame(y)
        y_label = y_label.iloc[:, 0].apply(lambda x: self.status_dict[int(x)])
        y_label = np.array(y_label)
        return y_label


if __name__ == '__main__':

    train = sys.argv[1]
    test = sys.argv[2]

    names = ['ID', 'Label']
    for i in range(30):
        names.append('f' + str(i + 1))
    df_train = pd.read_csv(train, names=names)

    LR = LogisticRegression(Lambda=0.01,lr=0.05)
    df_train = LR.pretreatment(df_train)
    LR.fit(df_train.iloc[:, 2:33], df_train.iloc[:, 33])

    df_test = pd.read_csv(test, names=names)
    df_test = LR.pretreatment(df_test, 'test')
    y_pre = LR.predict(df_test.iloc[:, 2:33])
    for i in y_pre:
        print(i)
