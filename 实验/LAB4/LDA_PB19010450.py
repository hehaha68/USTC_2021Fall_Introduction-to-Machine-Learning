# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/24 10:41
@Author  : 和泳毅
@FileName: LDA.py
@SoftWare: PyCharm
"""

import jieba
import jieba.posseg as psg
import pandas as pd
import numpy as np
import re
from numba import jit
import time

dic_file = "word/dict.txt"
stop_file = "word/stopwords.txt"


def word_cut(text):
    """
    将一个中文文本处理为词语列表
    :param text: 中文文本
    :return: 词语列表
    """
    jieba.load_userdict(dic_file)  # 可以在jieba词典中添加新词
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']  # 设定只需要名词、专有名词、动名词
    # 读取设定的停用词
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # 分词
    seg_list = psg.cut(text)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        find = 0
        # 不记录停用词以及小于两个字的词
        if word in stop_list or len(word) < 2:
            continue
        if seg_word.flag in flag_list:
            word_list.append(word)
    return word_list


def text_handling(data):
    """
    对n个中文文本处理，并统计总词语频数
    :param data: n个中文文本
    :return: 词语列表，词语频数
    """
    freq = {}
    word_list = []
    for i in range(len(data)):
        word_list.append([])
        word_list[-1] = word_cut(data[i])
        for word in word_list[i]:
            if word in freq.keys():
                freq[word] += 1
            else:
                freq[word] = 1
    return word_list, freq


class LDATopicModel:

    def __init__(self, data, topic_num, alpha, beta):
        """
        data: 所有文档
        text_num: 文档个数
        topic_num: 主题个数
        alpha: Dirichlet分布参数
        beta: Dirichlet分布参数
        """
        self.data = data
        self.text_num = len(data)
        self.topic_num = topic_num
        self.alpha = alpha
        self.beta = beta

    def preprocess(self, freq, drop_num=0):
        """
        去除高频词语，并将词语序列映射为词典索引序列（降低复杂度）
        :param freq: 词语频数
        :param drop_num: 需要去除的高频词语个数
        """
        # 合并所有文档的单词序列
        all_words = []
        for i in range(self.text_num):
            all_words = all_words + self.data[i]
        # 去除重复单词
        unique_words = list(set(all_words))
        # 提供去除高频词语功能
        unique_words.sort(key=lambda wd: freq[wd], reverse=True)
        self.word_list = unique_words[drop_num:]
        # 将词语映射为词典索引
        self.word_idx = {}
        for t, wd in enumerate(self.word_list):
            self.word_idx[wd] = t
        # 重新处理所有文档，将词语序列映射为词典索引序列
        self.word_map = []
        for word_line in self.data:
            self.word_map.append([])
            for word in word_line:
                if word in self.word_list:
                    self.word_map[-1].append(self.word_idx[word])

    def fit(self, epochs):
        """
        训练过程
        :param epochs: 迭代次数
        """
        word_num = len(self.word_list)
        # 设置超参
        alpha = self.alpha * np.ones(self.topic_num)
        beta = self.beta * np.ones(word_num)
        # 初始化参数
        nd = np.zeros((self.text_num, self.topic_num)).astype(np.int64)
        mk = np.zeros((self.topic_num, word_num)).astype(np.int64)
        z = [np.random.randint(0, self.topic_num, len(self.word_map[i])) for i in range(0, self.text_num)]
        new_z = [np.zeros(len(self.word_map[i])).astype(np.int64) for i in range(0, self.text_num)]
        # 统计变量
        for i, wordlist in enumerate(self.word_map):
            mk, nd = self.Gibbs_0(np.array(wordlist), mk, nd, i, np.array(z[i]))

        # 吉布斯采样
        for epoch in range(1, epochs + 1):
            if epoch % 5 == 0:
                print('--第', epoch, '次迭代--')

            temp = (np.sum(mk, axis=1) + np.sum(beta)) * (np.sum(nd) + np.sum(alpha))
            for i, wordlist in enumerate(self.word_map):
                new_z[i] = list(self.Gibbs_1(np.array(wordlist), temp, mk, beta, alpha, nd, i, np.array(new_z[i])))

            for i, wordlist in enumerate(self.word_map):
                mk, nd, z[i] = self.Gibbs_2(np.array(wordlist), mk, nd, np.array(z[i]), i, np.array(new_z[i]))

        # 参数输出
        self.phi = mk + np.broadcast_to(beta, (self.topic_num, word_num))
        self.phi = self.phi / np.transpose(np.broadcast_to(np.sum(self.phi, axis=1), (word_num, self.topic_num)))

        self.theta = nd + np.broadcast_to(alpha, (self.text_num, self.topic_num))
        self.theta = self.theta / np.transpose(
            np.broadcast_to(np.sum(self.theta, axis=1), (self.topic_num, self.text_num)))

    def topics_words(self, word_num):
        """
        输出每个主题下概率最高的的n个词语
        :param word_num: 需要输出每个主题下概率最高的的词语个数
        :return: 词语序列
        """
        topic_word = np.argsort(-self.phi, axis=1)
        top_words = [[]] * self.topic_num
        for i in range(self.topic_num):
            top_words[i] = []
            for j in range(word_num):
                top_words[i].append(self.word_list[topic_word[i][j]])
        return top_words

    def text_topics(self, topic_num):
        """
       输出每个文档下概率最高的的n个主题
        :param topic_num: 需要输出每个文档下概率最高的的主题个数
        :return: 主题序列
        """
        return np.argsort(-self.theta, axis=1)[:, :topic_num]

    @staticmethod
    @jit(nopython=True)
    def Gibbs_0(wordlist, mk, nd, i, z):
        for j, word in enumerate(wordlist):
            nd[i, z[j]] += 1
            mk[z[j], word] += 1
        return mk, nd

    @staticmethod
    @jit(nopython=True)
    def Gibbs_1(wordlist, temp, mk, beta, alpha, nd, i, new_z):
        for j, word in enumerate(wordlist):
            p = (mk[:, word] + beta[word]) * (nd[i, :] + alpha) / temp
            p = p / np.sum(p)
            if not (p > 0.).all():
                print(p)
            new_z[j] = np.argmax(np.random.multinomial(1, p))
        return new_z

    @staticmethod
    @jit(nopython=True)
    def Gibbs_2(wordlist, mk, nd, z, i, new_z):
        for j, word in enumerate(wordlist):
            mk[z[j], word] -= 1
            mk[new_z[j], word] += 1
            nd[i, z[j]] -= 1
            nd[i, new_z[j]] += 1
            z[j] = new_z[j]
        return mk, nd, z


if __name__ == '__main__':
    data = pd.read_excel("data.xlsx")
    word_list, freq = text_handling(data['content'])
    lda = LDATopicModel(data=word_list, topic_num=8, alpha=0.3, beta=0.3)
    start = time.time()
    lda.preprocess(freq=freq, drop_num=0)
    end1 = time.time()
    lda.fit(epochs=50)
    result = lda.topics_words(word_num=15)
    end2 = time.time()
    print("学习结束，预处理用时{}s, 迭代用时{}s。".format(end1 - start, end2 - end1))
    for i in range(8):
        print("Topic", i + 1, ":", result[i])
