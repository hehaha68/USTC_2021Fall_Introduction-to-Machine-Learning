# -*- coding:utf-8 -*-
"""
@Time    : 2021.12.5
@Author  : 和泳毅
@FileName: dpc.py
@SoftWare: Pycharm
"""

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import davies_bouldin_score
from numba import jit

plt.style.use('seaborn')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# -------- Density Peak Clustering --------
class DPC():
    """
    dis: 二维数组,两点之间的欧氏距离
    rho: 一维数组,局部密度。可选截断核(cut_off)或高斯核(gauss)方式计算
    delta: 一维数组,基于密度的距离
    sorted_rho: 一维数组,局部密度降序排列对应下标
    center_index: 簇中心下标
    label: 分类标签
    """

    def pretreatment(self, df):
        return np.array((df - df.min()) / (df.max() - df.min()))

    def fit(self, x, d_c, kernal="cut_off"):
        self.dis, self.rho, self.delta, self.sorted_rho = self._fit(x, d_c, kernal)
        self.plot_decesion()

    @staticmethod
    @jit(nopython=True)
    def _fit(x, d_c, kernal):
        n = x.shape[0]
        dis, rho, delta = np.zeros((n, n)), np.zeros(n), np.zeros(n)
        # 计算欧式距离与局部密度
        for i in range(n):
            for j in range(i + 1, n):
                dis[i][j] = dis[j][i] = np.linalg.norm(x[i] - x[j])
            if kernal == "gauss":
                rho[i] = np.exp(-(dis[i] / d_c) ** 2).sum() - 1
            elif kernal == 'cut_off':
                for j in range(n):
                    rho[i] += 1 if dis[i][j] < d_c else 0
            else:
                print("This kernal method is not supported.")
                return
        # 计算基于密度的距离
        index = np.argsort(-rho)
        delta[index[0]] = dis[index[0]].max()
        for i in range(1, n):
            delta[index[i]] = dis[index[i]][index[0:i]].min()
        return dis, rho, delta, index

    def predict(self, x, rho_value, delta_value):
        n = x.shape[0]
        self.label = np.zeros(n)
        self.center_index = []

        j = 1
        for i in range(n):
            # 聚类中心
            if self.rho[i] >= rho_value and self.delta[i] >= delta_value:
                self.center_index.append(i)
                self.label[i] = j
                j += 1
            # 离群点
            elif self.rho[i] < rho_value and self.delta[i] >= delta_value:
                self.label[i] = -1

        # 剩余点
        index = self.sorted_rho
        for i in range(n):
            if self.label[index[i]] == 0:
                j = np.argmin(self.dis[index[i]][index[:i]])
                self.label[index[i]] = self.label[index[j]]

    def plot_decesion(self):
        trace0 = go.Scatter(x=self.rho, y=self.delta, mode='markers', marker_size=6)
        layout = go.Layout(title='决策图', xaxis_title="ρ", yaxis_title="δ", width=600, height=600, template="plotly")
        data = [trace0]
        fig = go.Figure(data, layout)
        plotly.offline.iplot(fig)

    def plot_result(self, X):
        plt.figure(figsize=(10, 6), dpi=80)
        plt.scatter(x=X[:, 0], y=X[:, 1], c=self.label, s=70, cmap='tab20', alpha=0.5)
        plt.scatter(x=X[self.center_index][:, 0], y=X[self.center_index][:, 1], marker="x", s=100, c="r")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title('聚类结果', fontsize=15)
        plt.show()


if __name__ == '__main__':
    file = ['Datasets/D31.txt', 'Datasets/R15.txt', 'Datasets/Aggregation.txt', 'Datasets/spiral.txt']
    df = pd.read_csv(file[0], sep=" ", header=None)
    dpc = DPC()
    data = dpc.pretreatment(df)
    dpc.fit(data, d_c=0.06)
    dpc.predict(data, rho_value=84, delta_value=0.06)  # for D31 dc=0.06
    # dpc.predict(data, rho_value=37, delta_value=0.08)  # for R15 dc=0.06
    # dpc.predict(data, rho_value=29, delta_value=0.20)  # for Aggregation dc=0.075
    # dpc.predict(data, rho_value=16, delta_value=0.19)  # for spiral dc=0.1 with gauss kernel
    dpc.plot_result(data)
    print('DBI: {}'.format(davies_bouldin_score(df, dpc.label)))
