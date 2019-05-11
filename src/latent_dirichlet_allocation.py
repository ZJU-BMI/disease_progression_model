# encoding=utf-8
import numpy as np
import random
import math
import os
import datetime
import csv


class LatentDirichletAllocation(object):
    def __init__(self, data, num_topics, alpha, beta):
        """
        :param data: data structure
            level 1: sequence of docs [doc_1, doc_2, ..., doc_M]
            level 2: sequence of tokens [token_1, token_2, ... token_N_m]
        :param num_topics: number of  topics
        :param alpha: vector, prior of dirichlet distribution
        :param beta: vector, prior of dirichlet distribution

        Follow the notations in parameter estimation for text analysis
        """
        self.__data = data

        self.__alpha = alpha
        self.__beta = beta
        self.__K = num_topics
        self.__Phi = None
        self.__Theta = None
        self.__M = None
        self.__V = None
        self.__N = None

        self.__topic = None
        self.__n_z = None
        self.__n_w = None

        self.__iteration = None

        self.__initialization()

    def __initialization(self):
        # initialize M, V, N
        self.__M = len(self.__data)
        self.__N = [len(doc) for doc in self.__data]
        max_token_index = 0
        for i in range(len(self.__data)):
            for j in range(len(self.__data[i])):
                token_index = self.__data[i][j]
                if token_index > max_token_index:
                    max_token_index = token_index
        self.__V = max_token_index + 1

        # initialize parameters
        self.__Theta = np.random.uniform(0, 1, [self.__M, self.__K])
        self.__Phi = np.random.uniform(0, 1, [self.__K, self.__V])
        for i in range(len(self.__Theta)):
            row_sum = np.sum(self.__Theta[i])
            self.__Theta[i] = self.__Theta[i] / row_sum
        for i in range(len(self.__Phi)):
            row_sum = np.sum(self.__Phi[i])
            self.__Phi[i] = self.__Phi[i] / row_sum

        # initialize hidden state assignment
        self.__topic = list()
        for i in range(len(self.__data)):
            self.__topic.append(list())
            for j in range(len(self.__data[i])):
                self.__topic[-1].append(random.randint(0, self.__K - 1))

        # initialize count matrix
        self.__n_z = np.zeros([self.__M, self.__K])
        self.__n_w = np.zeros([self.__K, self.__V])
        for i in range(len(self.__data)):
            for j in range(len(self.__data[i])):
                token = self.__data[i][j]
                topic = self.__topic[i][j]
                self.__n_z[i][topic] += 1
                self.__n_w[topic][token] += 1
        print('initialized succeed')

    def optimization(self, iteration_num):
        self.__iteration = iteration_num

        for i in range(iteration_num):
            self.__inference()
            self.__parameter_updating()
            self.__perplexity()
        print('optimization process finished')

    def __inference(self):
        for i in range(len(self.__data)):
            for j in range(len(self.__data[i])):
                self.__sampling_hidden_variable(i, j)

    def __sampling_hidden_variable(self, i, j):
        pass

    def __parameter_updating(self):
        pass

    def __perplexity(self, test_data=None):
        """
        The code is right, but the probability decreases in an exponential manner, which will inevitably
        cause underflow problem
        :param test_data: if test data is not None, calculate the likelihood of test data
                          if test data is None, calculate the likelihood of training data
        :return:
        """
        pass

    def save_result(self, folder_path):
        pass


def synthetic_data_generator(num_document, max_length, min_length, unique_tokens):
    corpus = list()
    for i in range(num_document):
        corpus.append(list())
        doc_length = random.randint(min_length, max_length)
        for j in range(doc_length):
            corpus[-1].append(random.randint(0, unique_tokens - 1))
    return corpus


def unit_test():
    num_document = 30
    max_length = 10
    min_length = 4
    unique_tokens = 8
    document = synthetic_data_generator(num_document, max_length, min_length, unique_tokens)
    num_topics = 4
    alpha = [0.1 for _ in range(num_topics)]
    beta = [0.1 for _ in range(unique_tokens)]
    lda = LatentDirichletAllocation(document, alpha=alpha, beta=beta, num_topics=num_topics)
    lda.optimization(10)
    lda.save_result(os.path.abspath('../resource/'))


if __name__ == '__main__':
    unit_test()
