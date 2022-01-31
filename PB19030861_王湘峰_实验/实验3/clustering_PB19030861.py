import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as plt1
from sklearn.metrics import davies_bouldin_score as dbs
from numba import jit


@jit
def distance(x):
    m = x.shape[0]
    dis = np.zeros([m, m])
    # 距离矩阵
    for i in range(m - 1):
        for j in range(i + 1, m):
            dis[i, j] = np.sqrt(np.sum((x[i] - x[j]) ** 2))
            dis[j, i] = dis[i, j]
    return dis


class DPC:

    def __init__(self, dc, thp, thd):
        self.dc = dc
        self.thp = thp
        self.thd = thd
        self.delta = []
        self.rho = []
        self.center = []

    def process(self, x):
        # 归一化
        data = np.array((df - df.max()) / (df.max() - df.min()))
        # 计算距离
        self.dis = distance(data)
        # 计算密度
        for i in range(self.dis.shape[0]):
            rho = 0
            for j in range(self.dis.shape[0]):
                rho += 1 if self.dis[i][j] < self.dc else 0
            self.rho.append(rho)
        # 初始化delta(全部置为0)
        self.delta = [0] * x.shape[0]
        # 将点按照密度从大到小排列，排序后的索引记录在index中
        index = sorted(range(x.shape[0]), key=lambda i: self.rho[i], reverse=True)
        # 计算每个点的delta，全局密度最大的点的delta置为距离最大值
        self.delta[index[0]] = self.dis[index[0]].max()
        for i in range(1, x.shape[0]):
            # index[0:i]即是密度比点index[i]大的点的集合
            self.delta[index[i]] = self.dis[index[i]][index[0:i]].min()
        return data

    def cluster(self, x):
        # 保留原始数据方便后续画图
        self.x = x
        # 预处理数据
        self.process(x)
        # 初始化每个点的类别
        self.cate = [0] * x.shape[0]
        # 寻找聚类中心以及离群点，初始化类别数量cate=1
        cate = 1
        for i in range(x.shape[0]):
            # 聚类中心
            if self.rho[i] > self.thp and self.delta[i] > self.thd:
                self.center.append(i)
                self.cate[i] = cate
                cate += 1
            #离群点
            elif self.rho[i] < self.thp and self.delta[i] > self.thd:
                self.cate[i] = -1
        # 为每个点分配类别
        index = sorted(range(x.shape[0]), key=lambda i: self.rho[i], reverse=True)
        for i in range(x.shape[0]):
            if self.cate[index[i]] == 0:
                # 每个点的类别与比其密度高的点中最近的那一个相同
                j = np.argmin(self.dis[index[i]][index[:i]])
                self.cate[index[i]] = self.cate[index[j]]
        print('共有{}个类'.format(cate - 1))

    def decision(self):
        point = plt1.scatter(x=self.rho, y=self.delta)
        point.show()

    def show(self):
        plt.figure(figsize=(10, 6), dpi=80)
        plt.scatter(x=self.x[0], y=self.x[1], c=self.cate, marker='h')
        plt.scatter(x=self.x[0][self.center], y=self.x[1][self.center], marker='x', c='r')
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('Aggregation.txt', header=None, sep=' ')

    # D31
    # dc = 0.03
    # thp = 10
    # thd = 0.07

    # R15
    # dc = 0.05
    # thp = 10
    # thd = 0.1

    # Aggregation
    dc = 0.1
    thp = 10
    thd = 0.13

    f = DPC(dc, thp, thd)
    f.cluster(df)
    f.decision()
    f.show()
    print('DBI得分为：', dbs(df, f.cate))
