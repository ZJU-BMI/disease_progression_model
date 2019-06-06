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
            level 1: sequence of doc [doc_1, doc_2, ..., doc_M]
            level 2: For each doc i, the structure is : [token_i_1, token__i_2, ..., token_i_N_i]
            Note, each token is represented using a index in [0, token_size-1]
        :param num_topics: number of topics
        :param alpha: vector, prior of dirichlet distribution
        :param beta: vector, prior of dirichlet distribution
        """
        self.__data = data
        # num of doc
        self.__M = None
        # length of each doc
        self.__N = None
        # token size
        self.__V = None
        # num of topic
        self.__K = num_topics
        self.__alpha = alpha
        self.__beta = beta
        # doc topic count matrix
        self.__n_z = None
        # topic word count matrix
        self.__n_w = None
        # word topic assignment matrix
        self.__z = None
        self.__theta = None
        self.__phi = None
        self.__initialization()

    def __initialization(self):
        self.__M = len(self.__data)
        self.__N = [len(item) for item in self.__data]
        max_index = 0
        for i in range(self.__M):
            for j in range(self.__N[i]):
                token_index = self.__data[i][j]
                if token_index > max_index:
                    max_index = token_index
        self.__V = max_index + 1

        # topic initialization
        word_topic_assign = list()
        for i in range(self.__M):
            word_topic_assign.append(list())
            for _ in range(self.__N[i]):
                topic = random.randint(0, self.__K-1)
                word_topic_assign[-1].append(topic)
        self.__z = word_topic_assign

        # updating count
        self.__n_z = np.zeros([self.__M, self.__K])
        for i in range(self.__M):
            for j in range(self.__N[i]):
                self.__n_z[i, self.__z[i][j]] += 1

        self.__n_w = np.zeros([self.__K, self.__V])
        for i in range(self.__M):
            for j in range(self.__N[i]):
                self.__n_w[self.__z[i][j], self.__data[i][j]] += 1

        self.__theta = np.random.uniform([self.__M, self.__K])
        self.__phi = np.random.uniform([self.__K, self.__V])
        print('initialized succeed')

    def optimization(self, iteration_num):
        for iteration in range(iteration_num):
            for i in range(self.__M):
                for j in range(self.__N[i]):
                    self.__updating_hidden_variable_assignment(i, j)
            self.__parameter_estimation()
            self.__train_likelihood()

        print('optimization process finished')

    def __updating_hidden_variable_assignment(self, i, j):
        # decrease
        token = self.__data[i][j]
        current_topic = self.__z[i][j]
        self.__n_w[current_topic, token] -= 1
        self.__n_z[i, current_topic] -= 1
        # sampling
        assignment_prob = list()
        for topic in range(self.__K):
            part_1 = (self.__n_w[topic, token] + self.__beta[token]) / np.sum(self.__n_w[topic] + self.__beta)
            part_2 = (self.__n_z[i, topic] + self.__alpha[topic]) / (np.sum(self.__alpha+self.__n_z[i])-1)
            assignment_prob.append(part_1*part_2)
        assignment_prob = np.array(assignment_prob)/np.sum(assignment_prob)
        cumulative_prob = [0]
        for prob in assignment_prob:
            cumulative_prob.append(cumulative_prob[-1]+prob)
        prob = random.uniform(0, 1)
        new_topic = 1000000
        for k in range(len(cumulative_prob)):
            if prob < cumulative_prob[k]:
                new_topic = k-1
                break
        # increase
        self.__z[i][j] = new_topic
        self.__n_w[new_topic, token] += 1
        self.__n_z[i, new_topic] += 1

    def __parameter_estimation(self):
        phi = np.zeros([self.__K, self.__V])
        theta = np.zeros([self.__M, self.__K])
        for i in range(self.__K):
            sum_ = np.sum(self.__n_w[i] + self.__beta)
            for j in range(self.__V):
                phi[i, j] = (self.__n_w[i, j] + self.__beta[j]) / sum_
        for i in range(self.__M):
            sum_ = np.sum(self.__n_z[i] + self.__alpha)
            for j in range(self.__K):
                theta[i, j] = (self.__n_z[i, j] + self.__alpha[j]) / sum_
        self.__phi = phi
        self.__theta = theta

    def __train_likelihood(self):
        """
        在本文中，由于数据的特殊性（每个观察只会出现一次），导致并不特别适合计算测试集似然值
        因此只提供计算
        :return:
        """
        log_likelihood = 0
        for i in range(self.__M):
            for j in range(self.__N[i]):
                phi = self.__phi[:, self.__data[i][j]]
                theta = self.__theta[i]
                log_likelihood += math.log(np.sum(phi*theta))
        print("log_likelihood: {}".format(log_likelihood))
        return log_likelihood

    def save_result(self, folder_path):
        result_dict = dict()
        result_dict['M'] = self.__M
        result_dict['N'] = self.__N
        result_dict['K'] = self.__K
        result_dict['V'] = self.__V
        result_dict['alpha'] = self.__alpha
        result_dict['beta'] = self.__beta

        now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir(os.path.join(folder_path, now))
        folder_path = os.path.join(folder_path, now)
        with open(os.path.join(folder_path, 'result_parameter.csv'.format(now)), "w",
                  encoding='utf-8-sig', newline='') as f:
            basic_data = [['parameter', 'value']]
            for key in result_dict:
                basic_data.append([key, result_dict[key]])
            csv.writer(f).writerows(basic_data)

        result_dict['n_z'] = self.__n_z
        with open(os.path.join(folder_path, 'n_z.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__n_z)
        result_dict['n_w'] = self.__n_w
        with open(os.path.join(folder_path, 'n_r.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__n_w)

        result_dict['theta'] = self.__theta
        with open(os.path.join(folder_path, 'theta.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__theta)
        result_dict['phi'] = self.__phi
        with open(os.path.join(folder_path, 'phi.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__phi)

        with open(os.path.join(folder_path, 'hidden_state_assignment.csv'.format(now)),
                  "w", encoding='utf-8-sig', newline='') as f:
            hidden = self.__z
            hidden_matrix = list()
            hidden_matrix.append(['patient_index', 'visit_index', 'subtype', 'risk_tier'])
            for i in range(len(hidden)):
                for j in range(len(hidden[i])):
                    hidden_matrix.append([i, j, hidden[i][j]])
            csv.writer(f).writerows(hidden_matrix)


def synthetic_data_generator(num_document, max_length, min_length, unique_tokens):
    corpus = list()
    for i in range(num_document):
        corpus.append(list())
        doc_length = random.randint(min_length, max_length)
        for j in range(doc_length):
            corpus[-1].append(random.randint(0, unique_tokens - 1))
    return corpus


def unit_test():
    num_document = 400
    max_length = 100
    min_length = 50
    unique_tokens = 200
    document = synthetic_data_generator(num_document, max_length, min_length, unique_tokens)
    num_topics = 10
    alpha = [0.1 for _ in range(num_topics)]
    beta = [0.1 for _ in range(unique_tokens)]
    lda = LatentDirichletAllocation(document, alpha=alpha, beta=beta, num_topics=num_topics)
    lda.optimization(10)
    lda.save_result(os.path.abspath('../../resource/LDA'))


if __name__ == '__main__':
    unit_test()
