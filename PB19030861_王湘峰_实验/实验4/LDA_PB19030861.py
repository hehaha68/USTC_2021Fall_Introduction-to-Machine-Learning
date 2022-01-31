import numpy as np
import pandas as pd
import jieba
import jieba.posseg as psg
import re
from time import time


def word_cut(text, stopword_list):
    jieba.initialize()
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(text)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        find = 0
        if word in stop_list or len(word) < 2:
            continue
        if seg_word.flag in flag_list:
            word_list.append(word)
    return word_list


class LDATopicModel:
    # insignificant symbols
    meaningless_symbol_list = ['.', '$', '#', '[', ']', '(', ')', '|', '*', ':', '=', '/',
                               '>', '<', '+', '{', '}', ',', '?', '&', '-', '@', "'", '%', '^', '，',
                               '。', '《', '》', '：', '；', '“', '”', '‘', '？']

    def __init__(self, raw_texts, k_topics):
        self.raw_data = raw_texts
        self.k_topics = k_topics
        self.beta = None
        self.eta = None
        self.theta = None
        self.data_set_mapped = None
        self.vocabulary_list = None
        self.vocabulary_idx = None

    def preprocess(self, drop_n_freq=30):
        num_docs = len(self.raw_data)
        all_words = list()
        raw_docs = list()
        try:
            stopword_list = open(stop_file, encoding='utf-8').read().split('\n')
        except:
            stopword_list = []
        # Remove meaningless symbols, get all words of which length >= 2
        for i in range(num_docs):
            doc_text = self.raw_data[i]
            for s in LDATopicModel.meaningless_symbol_list:
                doc_text = doc_text.replace(s, '')
            raw_docs.append(list())

            for word in word_cut(doc_text, stopword_list):
                if len(word) >= 2:
                    raw_docs[-1].append(word)
            all_words.extend(raw_docs[-1])

        # Get global word frequency and sort, remove top n frequent words, which is likely to be meaningless.
        unique_words = list(set(all_words))
        word_frequency = dict()
        for wd in unique_words:
            word_frequency[wd] = 0
        for wd in all_words:
            word_frequency[wd] += 1
        unique_words.sort(key=lambda wd: word_frequency[wd], reverse=True)
        self.vocabulary_list = unique_words[drop_n_freq:]

        # Each word in the vocabulary will be mapped to an positive integer for convenience of indexing.
        # Thus to look up a certain word in the vocabulary takes O(1) time instead of O(log|V|)
        self.vocabulary_idx = dict()
        for t, wd in enumerate(self.vocabulary_list):
            self.vocabulary_idx[wd] = t
        vocabulary_set = set(self.vocabulary_list)

        # Map words in the documents to integers.
        # Append only words in the vocabulary set.
        self.data_set_mapped = []
        for raw_doc in raw_docs:
            self.data_set_mapped.append([])
            for word in raw_doc:
                if word in vocabulary_set:
                    self.data_set_mapped[-1].append(self.vocabulary_idx[word])

    def fit(self, num_iterations):
        """Fit after preprocessing."""
        num_docs = len(self.data_set_mapped)
        num_vocabulary = len(self.vocabulary_list)
        # Give parameter for Dirichlet distribution: alpha, eta
        alpha = 2 * np.ones(self.k_topics)
        eta = 0.01 * np.ones(num_vocabulary)

        # Randomly initialize
        nd = np.zeros((num_docs, self.k_topics), dtype=np.int64)
        mk = np.zeros((self.k_topics, num_vocabulary), dtype=np.int64)
        z = [np.random.randint(0, self.k_topics, len(self.data_set_mapped[i])) for i in range(0, num_docs)]
        new_z = [np.zeros(len(self.data_set_mapped[i]), dtype=np.int64) for i in range(0, num_docs)]
        # Initialize nd and mk
        for d, doc in enumerate(self.data_set_mapped):
            for j, word in enumerate(doc):
                nd[d, z[d][j]] += 1
                mk[z[d][j], word] += 1

        # Gibbs Sampling
        for iteration in range(num_iterations):
            print('Iteration %d' % (iteration + 1))
            denominator = np.sum(mk, axis=1) + np.sum(eta)
            # update z
            for d, doc in enumerate(self.data_set_mapped):
                for j, word in enumerate(doc):
                    prob_d = (mk[:, word] + eta[word]) * (nd[d, :] + alpha) / denominator
                    prob_d = prob_d / np.sum(prob_d)
                    if not (prob_d > 0.).all():
                        print(prob_d)
                    new_z[d][j] = np.argmax(np.random.multinomial(1, prob_d))
            # Update nd and mk
            for d, doc in enumerate(self.data_set_mapped):
                for j, word in enumerate(doc):
                    mk[z[d][j], word] -= 1
                    mk[new_z[d][j], word] += 1
                    nd[d, z[d][j]] -= 1
                    nd[d, new_z[d][j]] += 1
                    z[d][j] = new_z[d][j]

            # Generate theta and beta
        self.beta = mk + np.broadcast_to(eta, (self.k_topics, num_vocabulary))
        self.beta /= np.transpose(np.broadcast_to(np.sum(self.beta, axis=1), (num_vocabulary, self.k_topics)))

        self.theta = nd + np.broadcast_to(alpha, (num_docs, self.k_topics))
        self.theta /= np.transpose(np.broadcast_to(np.sum(self.theta, axis=1), (self.k_topics, num_docs)))

    def topics_words(self, n):
        """Return top n words of each topic"""
        topic_words_map = np.argsort(-self.beta, axis=1)
        top_words = [[] for i in range(self.k_topics)]
        for i in range(self.k_topics):
            for j in range(0, n):
                top_words[i].append(self.vocabulary_list[topic_words_map[i, j]])
        return top_words


if __name__ == '__main__':
    stop_file = 'stopwords.txt'
    raw_data = pd.read_excel('data.xlsx')
    model = LDATopicModel(raw_texts=raw_data['content'], k_topics=8)
    start = time()
    model.preprocess(drop_n_freq=0)
    end = time()
    print('预处理用时{}s'.format(end - start))
    start = time()
    model.fit(num_iterations=60)
    end = time()
    print('训练用时{}s'.format(end - start))
    for i, words in enumerate(model.topics_words(n=15)):
        print('Topic %d:' % (i + 1), end='')
        for word in words:
            print(word, end=' ')
        print('')
    print('真实的主题有：体育 娱乐 彩票 房产 教育 游戏 科技 股票')
