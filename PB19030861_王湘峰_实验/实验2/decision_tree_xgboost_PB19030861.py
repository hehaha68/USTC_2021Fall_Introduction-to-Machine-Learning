import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def Obj(data, lamda, gamma):
    G = -2 * np.sum(data[:, -2] - data[:, -1])
    H = 2 * data.shape[0]
    return -0.5 * (G ** 2 / (H + lamda)) + gamma


def NodeSplit(data, feature, value):
    left = data[np.nonzero(data[:, feature] <= value)[0], :]
    right = data[np.nonzero(data[:, feature] > value)[0], :]
    return right, left


def Leaf_weight(data, lamda):
    # 返回叶节点的值
    return -(-2 * np.sum(data[:, -2] - data[:, -1])) / (2 * data.shape[0] + lamda)


# 二元切分
def chooseBestSplit(data, gamma, lamda, depth=1):
    min_gain = thresh[0]  # 允许的最小信息增益,小于则直接创建叶节点
    min_num = thresh[1]  # 切分的最小样本数

    # 若所有特征值都相同，停止切分
    if len(set(data[:, -2].T.tolist())) == 1:
        return None, Leaf_weight(data, lamda)

    # 若样本数目小于阈值，则停止切分并生成叶节点
    if np.shape(data)[0] < min_num:
        return None, Leaf_weight(data, lamda)

    m, n = np.shape(data)
    parent = Obj(data, lamda, gamma)  # 计算划分前该节点的得分
    max_gain = float("-inf")
    bestfeature = 0
    bestValue = 0

    for feature in range(n - 2):  # 遍历数据的每个属性特征
        for splitVal in set((data[:, feature].T.tolist())):  # 遍历每个特征里不同的特征值
            right, left = NodeSplit(data, feature, splitVal)  # 对每个特征进行二元分类
            if (np.shape(right)[0] < 1) or (np.shape(left)[0] < 1):
                continue
            child = Obj(right, lamda, gamma) + Obj(left, lamda, gamma)
            score = parent - child
            if score > max_gain:
                bestfeature = feature
                bestValue = splitVal
                max_gain = score

    # 如果切分后的增益小于阈值，直接创建叶节点
    if max_gain < min_gain or depth > maxdepth:
        return None, Leaf_weight(data, lamda)

    return bestfeature, bestValue  # 返回特征编号和用于切分的特征值


# 构建tree
def createTree(data, gamma, lamda, depth=1):
    # 建树
    dept = depth
    feature, value = chooseBestSplit(data, gamma, lamda, dept)
    if feature == None:
        return value  # 满足停止条件时返回叶结点值
    # 切分后赋值
    DecTree = {}
    DecTree['feature'] = feature
    DecTree['values'] = value
    # 切分后的左右子树
    left, right, = NodeSplit(data, feature, value)
    dept += 1
    DecTree['left'] = createTree(left, gamma, lamda, dept)
    DecTree['right'] = createTree(right, gamma, lamda, dept)
    return DecTree


def get_w(x, Tree):
    if type(Tree) == dict:
        if x[Tree['feature']] > Tree['values']:
            return get_w(x, Tree['left'])
        else:
            return get_w(x, Tree['right'])
    else:
        return Tree


def fit(train_data, tree_num, gamma, lamb):
    f = []
    RMSE = []
    R2 = []
    for j in range(tree_num):
        f.append(createTree(train_data, gamma=gamma, lamda=lamb, depth=1))
        for i in range(train_data.shape[0]):
            train_data[i, -1] += get_w(train_data[i, :], f[j]) * lr

        # print("正在训练第个{}棵决策树".format(j + 1))
        rmse = (np.sum((train_data[:, -2] - train_data[:, -1]) ** 2) / m) ** 0.5
        RMSE.append(rmse)
        # print("训练集RMSE:", rmse)
        r2 = 1 - rmse ** 2 / np.var(train_data[:, -2])
        R2.append(r2)
        # print('训练集R2:', r2)
    return f, RMSE, R2


def predict(test_data, f):
    for j in range(tree_num):
        for i in range(test_data.shape[0]):
            test_data[i, -1] += get_w(test_data[i, :], f[j]) * lr

    # rmse = (np.sum((test_data[:, -2] - test_data[:, -1]) ** 2) / test_data.shape[0]) ** 0.5
    # print("测试集RMSE:", rmse)
    # r2 = 1 - rmse ** 2 / np.var(test_data[:, -2])
    # print('测试集R2', r2)


if __name__ == '__main__':
    # 初始化训练集和测试集
    raw = pd.read_csv('train.data', header=None)
    raw[41] = np.zeros(raw.shape[0])
    df1 = raw.sample(frac=0.8)
    df2 = raw[~raw.index.isin(df1.index)]

    '''df1 = pd.read_csv('train.data', header=None)
    df2 = pd.read_csv('ailerons.test', header=None)
    df1[41] = np.zeros(df1.shape[0])
    df2[41] = np.zeros(df2.shape[0])'''

    train_data = np.array(df1)
    test_data = np.array(df2)
    m, n = train_data.shape

    # 参数部分
    maxdepth = 3
    tree_num = 25
    gamma = 0
    lamb = 1
    lr = 0.2
    thresh = (0, 3)

    # 训练与预测部分
    f, RMSE, R2 = fit(train_data, tree_num, gamma, lamb)
    predict(test_data, f)

    # 绘图部分
    '''
    x = np.arange(1, len(MSE) + 1)
    plt.xlabel('Iteration times')
    plt.ylabel('MSE')
    plt.plot(x, MSE)
    plt.show()
    plt.xlabel('Iteration times')
    plt.ylabel('R2')
    plt.plot(x, R2)
    plt.show()
    err = test_data[:, -1] - test_data[:, -2]
    err.tolist()
    x = np.arange(1, len(err) + 1)
    plt.xlabel('Error of every sample')
    plt.ylabel('ERROR')
    plt.plot(x, err)
    plt.show()'''
