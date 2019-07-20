# encoding=utf-8
import numpy as np
import random
import os
import datetime
import csv
from math import log, exp
import data_reader


class DiseaseProgressionModel(object):
    def __init__(self, data, num_risk_tier, num_subtype, alpha, beta, gamma, dependence_constraint=2):
        """
        :param data: data structure
            level 1: sequence of patient trajectories [trajectory_1, trajectory_2, ..., trajectory_M]
            level 2: For each trajectory of patient i, the structure is : [visit_1, visit_2, ..., visit_T_i]
            level 3: for j th visit of i th patient, the structure is: [observation_1, ..., observation_ij]
            Note, each observation is represented using a index in [0, num_unique_type_observation-1]
            and each type of observation exists at most once in a single visit
        :param num_risk_tier: number of  risk tier
        :param num_subtype: number of subtypes of patients
        :param alpha: vector, prior of dirichlet distribution
        :param beta: vector, prior of dirichlet distribution
        :param gamma: vector, prior of dirichlet distribution
        """
        self.__data = data

        # all notations follow Table 1
        # the variables will be initialized in initialization function
        self.__M = None
        self.__T = None
        self.__N = None
        self.__o = None
        self.__R = num_risk_tier
        self.__K = num_subtype
        self.__O = None
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__n_z = None
        self.__n_r = None
        self.__n_o = None
        self.__Theta = None
        self.__Pi = None
        self.__Phi = None
        self.__iteration = None
        self.__hidden_state_assignment = None
        self.__dependence_constraint = dependence_constraint

        # initialize all variables
        self.__initialization()

    def __initialization(self):
        # set parameters
        self.__M = len(self.__data)
        self.__T = [len(trajectory) for trajectory in self.__data]
        self.__N = list()
        for trajectory in self.__data:
            self.__N.append([len(sequence) for sequence in trajectory])
        self.__O = 0
        for trajectory in self.__data:
            for sequence in trajectory:
                for item in sequence:
                    if item > self.__O:
                        self.__O = item
        self.__O += 1

        # initialize parameter
        # note all parameters follows categorical distribution, i.e., the sum of each row (2D case) should be 1
        # note the markov chain needs a initial state, so the size of state space is K+1
        # we define the initial state is K
        self.__Pi = np.random.uniform(0, 1, [self.__K+1, self.__R])
        self.__Phi = np.random.uniform(0, 1, [self.__K+1, self.__R, self.__O])
        # normalization
        for i in range(len(self.__Pi)):
            row_sum = np.sum(self.__Pi[i])
            self.__Pi[i] = self.__Pi[i]/row_sum
        for i in range(len(self.__Phi)):
            for j in range(len(self.__Phi[i])):
                row_sum = np.sum(self.__Phi[i][j])
                self.__Phi[i][j] = self.__Phi[i][j]/row_sum

        # initialize Theta with constraint
        self.__Theta = np.zeros([self.__K + 1, self.__K + 1])
        constraint = self.__dependence_constraint
        init_trans_distribution = np.random.dirichlet(np.full(self.__K, self.__alpha))
        for i in range(self.__K):
            self.__Theta[self.__K, i] = init_trans_distribution[i]
        for i in range(self.__K):
            if i+constraint < self.__K:
                init_trans_distribution = np.random.dirichlet(np.full(constraint+1, self.__alpha))
                for j in range(i, i+constraint+1):
                    self.__Theta[i, j] = init_trans_distribution[j-i]
            else:
                init_trans_distribution = np.random.dirichlet(np.full(self.__K-i, self.__alpha))
                for j in range(i, i+self.__K-i):
                    self.__Theta[i, j] = init_trans_distribution[j-i]

        # initialize topic assignment with constraint
        # the index of initial state is K
        self.__n_z = np.zeros([self.__K+1, self.__K+1])
        self.__n_r = np.zeros([self.__K+1, self.__R])
        self.__n_o = np.zeros([self.__K+1, self.__R, self.__O])
        self.__hidden_state_assignment = list()
        # assignment
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            self.__hidden_state_assignment.append(list())
            for j in range(len(trajectory)):
                if j == 0:
                    current_subtype = random.randint(0, self.__K-1)
                else:
                    # the latter state can't be better than the previous one
                    current_subtype = random.randint(self.__hidden_state_assignment[-1][-1][0], self.__K-1)
                risk_tier = random.randint(0, self.__R-1)
                self.__hidden_state_assignment[-1].append([current_subtype, risk_tier])

        # updating count
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            for j in range(len(trajectory)):
                # updating n_z
                current_subtype = self.__hidden_state_assignment[i][j][0]
                if j == 0:
                    previous_subtype = self.__K
                else:
                    previous_subtype = self.__hidden_state_assignment[i][j-1][0]
                self.__n_z[previous_subtype, current_subtype] += 1

                # updating n_r
                risk_tier = self.__hidden_state_assignment[i][j][1]
                self.__n_r[current_subtype, risk_tier] += 1

                # updating n_o
                sequence = trajectory[j]
                for k in range(len(sequence)):
                    observation = sequence[k]
                    self.__n_o[current_subtype, risk_tier, observation] += 1
        print('initialize procedure accomplished')

    def optimization(self, iteration_num):
        for i in range(iteration_num):
            self.__iteration = iteration_num
            start_time = datetime.datetime.now()
            self.__inference()
            self.__parameter_updating()
            likelihood = self.log_likelihood()
            end_time = datetime.datetime.now()
            print('iteration {} accomplished, current log-likelihood {}, time cost {}'.
                  format(i+1, likelihood, end_time-start_time))

    def __inference(self):
        # updating z
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            for j in range(len(trajectory)):
                self.__sampling_subtype(i, j)
        # updating r
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            for j in range(len(trajectory)):
                self.__sampling_risk_tier(i, j)

    def __sampling_subtype(self, i, j):
        # sampling new subtype with constraint
        no_next_indicator = -10000
        # Step 1: decrease count
        current_subtype, risk_tier = self.__hidden_state_assignment[i][j]
        if j == 0:
            previous_subtype = self.__K
            next_subtype = self.__hidden_state_assignment[i][j+1][0]
        elif j == len(self.__hidden_state_assignment[i])-1:
            previous_subtype = self.__hidden_state_assignment[i][j-1][0]
            # in fact, the final state doesn't have descendant state
            next_subtype = no_next_indicator
        else:
            previous_subtype = self.__hidden_state_assignment[i][j-1][0]
            next_subtype = self.__hidden_state_assignment[i][j+1][0]
        # minus hidden state count
        self.__n_z[previous_subtype][current_subtype] -= 1
        if next_subtype != no_next_indicator:
            self.__n_z[current_subtype][next_subtype] -= 1
        self.__n_r[current_subtype][risk_tier] -= 1
        observation_list = self.__data[i][j]
        for observation in observation_list:
            self.__n_o[current_subtype][risk_tier][observation] -= 1

        # Step 2: calculate probability with constraint
        # for part 1, follow equation 21
        # last visit check
        if j == 0:
            prev_state = self.__K
        else:
            prev_state = self.__hidden_state_assignment[i][j - 1][0]

        if prev_state == self.__K:
            candidate_state_count = self.__K
        else:
            candidate_state_count = self.__K-prev_state
            if candidate_state_count > self.__dependence_constraint + 1:
                candidate_state_count = self.__dependence_constraint + 1

        part_1 = np.zeros([candidate_state_count])
        if j == len(self.__N[i])-1:
            for k in range(prev_state, prev_state+candidate_state_count):
                numerator = self.__alpha + self.__n_z[prev_state][k]
                denominator = self.__alpha * candidate_state_count + np.sum(self.__n_z[prev_state])
                part_1[k-prev_state] = numerator/denominator
        else:
            next_state = self.__hidden_state_assignment[i][j + 1][0]
            if j == 0:
                for k in range(candidate_state_count):
                    numerator_1 = self.__alpha + self.__n_z[prev_state][k]
                    numerator_2 = self.__alpha + self.__n_z[k][next_state]
                    if k == next_state and k == prev_state:
                        numerator_2 += 1
                    denominator_1 = self.__alpha * candidate_state_count + np.sum(self.__n_z[prev_state])
                    denominator_2 = self.__alpha * candidate_state_count + np.sum(self.__n_z[k])
                    if prev_state == k:
                        denominator_2 += 1
                    part_1[k] = (numerator_1*numerator_2) / (denominator_1*denominator_2)
            else:
                for k in range(prev_state, prev_state + candidate_state_count):
                    numerator_1 = self.__alpha + self.__n_z[prev_state][k]
                    numerator_2 = self.__alpha + self.__n_z[k][next_state]
                    if k == next_state and k == prev_state:
                        numerator_2 += 1
                    denominator_1 = self.__alpha * candidate_state_count + np.sum(self.__n_z[prev_state])
                    denominator_2 = self.__alpha * candidate_state_count + np.sum(self.__n_z[k])
                    if prev_state == k:
                        denominator_2 += 1
                    part_1[k - prev_state] = (numerator_1*numerator_2) / (denominator_1*denominator_2)

        # for part 2, follow equation 18
        part_2 = np.zeros([candidate_state_count])
        if j == 0:
            for k in range(candidate_state_count):
                numerator = self.__gamma + self.__n_r[k][self.__hidden_state_assignment[i][j][1]]
                denominator = self.__gamma * self.__R + np.sum(self.__n_r[k])
                part_2[k] = numerator / denominator
        else:
            for k in range(prev_state, prev_state + candidate_state_count):
                numerator = self.__gamma + self.__n_r[k][self.__hidden_state_assignment[i][j][1]]
                denominator = self.__gamma * self.__R + np.sum(self.__n_r[k])
                part_2[k - prev_state] = numerator / denominator

        # for part 3, follow equation 20
        part_3 = np.zeros([candidate_state_count])
        if j == 0:
            for k in range(candidate_state_count):
                part_3_obs = 1
                risk_tier = self.__hidden_state_assignment[i][j][1]
                denominator = self.__O * self.__beta + np.sum(self.__n_o[k, risk_tier])
                for obs in self.__data[i][j]:
                    numerator = self.__beta + self.__n_o[k, risk_tier, obs]
                    part_3_obs = part_3_obs * numerator / denominator
                part_3[k] = part_3_obs
        else:
            for k in range(prev_state, prev_state + candidate_state_count):
                part_3_obs = 1
                risk_tier = self.__hidden_state_assignment[i][j][1]
                denominator = self.__O * self.__beta + np.sum(self.__n_o[k, risk_tier])
                for obs in self.__data[i][j]:
                    numerator = self.__beta + self.__n_o[k, risk_tier, obs]
                    part_3_obs = part_3_obs * numerator / denominator
                part_3[k - prev_state] = part_3_obs

        # normalization
        unnormalized_probability = part_1 * part_2 * part_3
        normalized_probability = unnormalized_probability / np.sum(unnormalized_probability)
        cumulative_probability = np.zeros(normalized_probability.shape)
        for a in range(len(cumulative_probability)):
            for b in range(a+1):
                cumulative_probability[a] += normalized_probability[b]

        # Step 3: sampling new assignment
        random_number = random.uniform(0, 1)
        new_subtype_sample = -10000
        for a in range(0, len(cumulative_probability)):
            if random_number <= cumulative_probability[a]:
                new_subtype_sample = a
                break
        if new_subtype_sample == -10000:
            raise ValueError('')
        if j != 0:
            new_subtype_sample = new_subtype_sample + self.__hidden_state_assignment[i][j-1][0]

        # Step 4: recover count
        self.__n_z[previous_subtype][new_subtype_sample] += 1
        if next_subtype != no_next_indicator:
            self.__n_z[new_subtype_sample][next_subtype] += 1
        self.__n_r[new_subtype_sample][risk_tier] += 1
        for observation in observation_list:
            self.__n_o[new_subtype_sample][risk_tier][observation] += 1
        self.__hidden_state_assignment[i][j][0] = new_subtype_sample

    def __sampling_risk_tier(self, i, j):
        # Step 1: decrease count
        current_subtype, current_risk_tier = self.__hidden_state_assignment[i][j]
        self.__n_r[current_subtype][current_risk_tier] -= 1
        observation_list = self.__data[i][j]
        for observation in observation_list:
            self.__n_o[current_subtype][current_risk_tier][observation] -= 1

        # Step 2: calculate probability
        # follow Equation 23.
        # For Part 1
        part_1_numerator = self.__gamma + self.__n_r[current_subtype]
        part_1_denominator = np.sum(self.__gamma + self.__n_r[current_subtype])
        part_1 = part_1_numerator / part_1_denominator
        # For Part 2
        part_2 = 1
        part_2_denominator = self.__O * self.__beta + np.sum(self.__n_o[current_subtype, :, :], axis=1)
        for observation in self.__data[i][j]:
            part_2_numerator = self.__beta + self.__n_o[current_subtype, :, observation]
            part_2 = part_2 * part_2_numerator / part_2_denominator

        unnormalized_probability = part_1 * part_2
        normalized_probability = unnormalized_probability / np.sum(unnormalized_probability)
        cumulative_probability = np.zeros(normalized_probability.shape)
        for k in range(len(cumulative_probability)):
            for l in range(k+1):
                cumulative_probability[k] += normalized_probability[l]

        # Step 3: sampling new assignment
        random_number = random.uniform(0, 1)
        new_risk_tier = -1
        for k in range(0, len(cumulative_probability)):
            if random_number <= cumulative_probability[k]:
                new_risk_tier = k
                break
        if new_risk_tier == -1:
            raise ValueError('')

        # Step 4: recover count
        self.__n_r[current_subtype][new_risk_tier] += 1
        for observation in self.__data[i][j]:
            self.__n_o[current_subtype][new_risk_tier][observation] += 1
        self.__hidden_state_assignment[i][j][1] = new_risk_tier

    def __parameter_updating(self):
        # updating Theta with constraint
        for i in range(self.__K + 1):
            if i == self.__K:
                c = self.__K
            elif i <= self.__K-self.__dependence_constraint-1:
                c = self.__dependence_constraint+1
            else:
                c = self.__K - i

            for j in range(i, self.__K):
                if j-i <= self.__dependence_constraint:
                    self.__Theta[i][j] = (self.__n_z[i][j] + self.__alpha) / (np.sum(self.__n_z[i]) + self.__alpha*c)
                    if self.__Theta[i][j] < 0:
                        print('ERROR')
        # updating Pi
        for i in range(self.__K + 1):
            for j in range(self.__R):
                self.__Pi[i][j] = (self.__n_r[i][j] + self.__gamma) / np.sum(self.__n_r[i] + self.__gamma)
                if self.__Pi[i][j] < 0:
                    print('ERROR')
        # updating Phi
        for i in range(self.__K + 1):
            for j in range(self.__R):
                for k in range(self.__O):
                    self.__Phi[i][j][k] = \
                        (self.__n_o[i][j][k] + self.__beta) / np.sum(self.__n_o[i][j] + self.__beta)
                    if self.__Phi[i][j][k] < 0:
                        print('ERROR')

    def log_likelihood(self, test_data=None):
        """
        The code is right, but the probability decreases in an exponential manner, which will inevitably
        cause underflow problem
        :param test_data: if test data is not None, calculate the likelihood of test data
                          if test data is None, calculate the likelihood of training data
        :return:
        """

        # calculate log s
        def calculate_part_1(cache):
            log_s = 0
            # note in this case, we don't need do care about the initial state, i.e., l = self.__K
            for k in range(self.__K):
                current_sum = cache[k]
                if k == 1:
                    log_s = current_sum
                else:
                    if log_s > current_sum:
                        log_s = log_s + log(1 + exp(current_sum - log_s))
                    else:
                        log_s = current_sum + log(1 + exp(log_s - current_sum))
            return log_s

        if test_data is None:
            data = self.__data
        else:
            data = test_data

        # cache initialization
        forward_cache = list()
        for trajectory in data:
            a_last_prob = self.__forward_procedure(trajectory, len(trajectory))
            forward_cache.append(a_last_prob)

        likelihood = 0
        for patient_index in range(len(forward_cache)):
            likelihood += calculate_part_1(forward_cache[patient_index])
        return likelihood

    # calculate log p given j, k
    def __calculate_part_3(self, trajectory, visit_id, subtype):
        j, k = visit_id, subtype
        log_p = 0
        for m in range(self.__R):
            current_sum = log(self.__Pi[k, m])
            for n in range(len(trajectory[j])):
                current_sum += log(self.__Phi[k, m, trajectory[j][n]])

            if m == 0:
                log_p = current_sum
            else:
                if log_p > current_sum:
                    log_p = log_p + log(1 + exp(current_sum - log_p))
                else:
                    log_p = current_sum + log(1 + exp(log_p - current_sum))
        return log_p

    # calculate log q with constraint
    def __forward_calculate_part_2(self, trajectory, cache, visit_id, subtype):
        j, k = visit_id, subtype
        log_q = 0

        # note in this case, we don't need do care about the initial state, i.e., l = self.__K
        # with constraint to the transition matrix between hidden states, we should skip some impossible
        # forward procedure path
        for l_ in range(self.__K):
            # constraint
            if self.__Theta[l_, k] == 0:
                continue

            log_p = self.__calculate_part_3(trajectory, j, k)
            if j == 0:
                current_sum = log(self.__Theta[self.__K, k]) + log_p
            else:
                current_sum = cache[-1][l_] + log(self.__Theta[l_, k]) + log_p

            if log_q == 0:
                log_q = current_sum
            else:
                if log_q > current_sum:
                    log_q = log_q + log(1 + exp(current_sum - log_q))
                else:
                    log_q = current_sum + log(1 + exp(log_q - current_sum))

            if log_q > 0:
                continue
        return log_q

    def __backward_calculate_part_2(self, trajectory, cache, visit_id, subtype):
        j, k = visit_id, subtype
        log_q = 0

        if j == len(trajectory):
            return 1

        # forward procedure path
        for l_ in range(self.__K):
            # constraint
            if self.__Theta[k, l_] == 0:
                continue

            log_p = self.__calculate_part_3(trajectory, j, l_)

            current_sum = cache[-1][l_] + log(self.__Theta[k, l_]) + log_p

            if log_q == 0:
                log_q = current_sum
            else:
                if log_q > current_sum:
                    log_q = log_q + log(1 + exp(current_sum - log_q))
                else:
                    log_q = current_sum + log(1 + exp(log_q - current_sum))

            if log_q > 0:
                continue
        return log_q

    def __forward_procedure(self, trajectory, terminate_idx):
        forward_cache = list()
        for visit_index in range(terminate_idx):
            a_ij = list()
            for patient_subtype in range(self.__K):
                a_ijk = self.__forward_calculate_part_2(trajectory, forward_cache, visit_index, patient_subtype)
                a_ij.append(a_ijk)
            forward_cache.append(a_ij)
        return forward_cache[-1]

    def __backward_procedure(self, trajectory, start_idx):
        backward_cache = list()
        for visit_index in range(start_idx, len(trajectory)+1).__reversed__():
            a_ij = list()
            for patient_subtype in range(self.__K):
                a_ijk = self.__backward_calculate_part_2(trajectory, backward_cache, visit_index, patient_subtype)
                a_ij.append(a_ijk)
            backward_cache.append(a_ij)
        return backward_cache[-1]

    def disease_state_assignment(self, new_doc):
        """
        Inference the hidden states of a given trajectory list set using Viterbi Algorithm,
        which was introduced in the section 3.B of 'A tutorial on hidden markov models and selected
        applications in speech recognition', and we follow the notations used in this article.
        :param new_doc: The data structure of 'new_doc' is same as the corresponding parameter of constructor:
        :return: hidden state estimation:
            level 1: sequence of patient trajectories [trajectory_1, trajectory_2, ..., trajectory_M]
            level 2: For each trajectory of patient i, the structure is : [state_1, state_2, ..., state_T_i]
        """

        def viterbi_algorithm(single_trajectory_list, theta, phi, pi):
            # delta indicates the maximum probability
            num_state = self.__K
            num_risk = self.__R
            delta = np.zeros([len(single_trajectory_list), num_state])
            psi = np.zeros([len(single_trajectory_list), num_state])

            for t, single_visit in enumerate(single_trajectory_list):
                # to avoid the underflow problem, we use the log likelihood

                # calculate the observation probability
                observation_log_prob_mat = np.zeros([num_state, num_risk])
                for j in range(num_state):
                    for r in range(num_risk):
                        observation_log_prob = 0
                        for observation in single_trajectory_list[t]:
                            observation_log_prob += log(phi[j, r, observation])
                        observation_log_prob_mat[j, r] = observation_log_prob

                # init step
                if t == 0:
                    prev_state = num_state
                    for j in range(num_state):
                        log_prob = 0
                        for r in range(num_risk):
                            if r == 0:
                                log_prob = log(theta[prev_state, j]) + log(pi[j, r]) + observation_log_prob_mat[j, r]
                            else:
                                temp = log(theta[prev_state, j]) + log(pi[j, r]) + observation_log_prob_mat[j, r]
                                if log_prob > temp:
                                    log_prob = log_prob + log(1 + exp(temp - log_prob))
                                else:
                                    log_prob = temp + log(1 + exp(log_prob - temp))
                        delta[t, j] = log_prob
                        psi[t, j] = -10000
                    continue

                # recursion step
                for j in range(num_state):
                    max_prob = -float('inf')
                    for prev_state in range(num_state):
                        log_prob = 0

                        # To avoid the numerical unstable problem when we constraint the state transition direction
                        if theta[prev_state, j] == 0:
                            transition_prob = 10**-50
                        else:
                            transition_prob = theta[prev_state, j]

                        for r in range(num_risk):
                            if r == 0:
                                log_prob = delta[t-1, prev_state] + log(transition_prob) + log(pi[j, r]) + \
                                           observation_log_prob_mat[j, r]
                            else:
                                temp = delta[t-1, prev_state] + log(transition_prob) + log(pi[j, r]) + \
                                       observation_log_prob_mat[j, r]
                                if log_prob > temp:
                                    log_prob = log_prob + log(1 + exp(temp - log_prob))
                                else:
                                    log_prob = temp + log(1 + exp(log_prob - temp))
                        if log_prob > max_prob:
                            max_prob = log_prob
                            delta[t, j] = log_prob
                            psi[t, j] = prev_state

            # terminate step
            estimated_state_list = list()
            estimated_last_state = np.argmax(delta[-1])
            estimated_state_list.insert(0, estimated_last_state)
            for i in range(len(single_trajectory_list)-1):
                estimated_last_state = int(psi[len(single_trajectory_list)-i-1, estimated_last_state])
                estimated_state_list.insert(0, estimated_last_state)
            return estimated_state_list

        theta_, phi_, pi_ = self.__Theta, self.__Phi, self.__Pi
        hidden_state_list = list()
        for trajectory in new_doc:
            single_hidden_state_list = viterbi_algorithm(trajectory, theta_, phi_, pi_)
            hidden_state_list.append(single_hidden_state_list)
        return hidden_state_list

    def risk_tier_inference(self, document, visit_idx=-1):
        """
        :param document:
        :param visit_idx:  visit index of target risk tier
        Note, if we have a trajectory whose length is 10, and
        we want to predict the risk tier of fifth visit, actually we have two candidate inference methods.
        1. The analysis mode, which assumes we can observe the entire trajectory, and using forward backward procedure
        to estimate the risk tier of fifth visit. 2. The predictive mode, which assumes we can only observe the start 4
        visits to estimate the risk tier of fifth visit via forward procedure.
        In our research, we use the analysis mode to estimate the risk tier
        :return:
        """
        pi = self.__Pi
        phi = self.__Phi
        risk_tier_list = list()
        for trajectory in document:
            # hidden state inference procedure
            if visit_idx == -1:
                idx = len(trajectory)-1
            else:
                idx = visit_idx
            log_forward_prob = np.array(self.__forward_procedure(trajectory, idx))
            log_backward_prob = np.array(self.__backward_procedure(trajectory, idx))
            numerator = log_forward_prob + log_backward_prob
            denominator = 0
            for i, _ in enumerate(log_forward_prob):
                current_sum = log_forward_prob[i] + log_backward_prob[i]
                if i == 0:
                    denominator = current_sum
                else:
                    if denominator > current_sum:
                        denominator = denominator + log(1 + exp(current_sum - denominator))
                    else:
                        denominator = current_sum + log(1 + exp(denominator - current_sum))
            hidden_state_prob_list = np.exp(numerator - denominator)

            risk_tier_prob = np.zeros([self.__R])
            for risk_tier in range(self.__R):
                prob_sum = 0
                for hidden_state in range(self.__K):
                    hidden_state_prob = hidden_state_prob_list[hidden_state]
                    # bayes numerator
                    observation_prob = 1
                    for n in range(len(trajectory[idx])):
                        observation_prob *= phi[hidden_state, risk_tier, trajectory[idx][n]]
                    numerator = pi[hidden_state, risk_tier] * observation_prob

                    # bayes denominator
                    denominator = 0
                    for r in range(self.__R):
                        denominator_obs = 1
                        for n_ in range(len(trajectory[idx])):
                            denominator_obs *= phi[hidden_state, risk_tier, trajectory[idx][n_]]
                        denominator_r = denominator_obs * pi[hidden_state, r]
                        denominator += denominator_r
                    prob_sum += numerator/denominator*hidden_state_prob
                risk_tier_prob[risk_tier] = prob_sum
            risk_tier_list.append(risk_tier_prob)
        return risk_tier_list

    def save_result(self, folder_path, index_name_dict, index_patient_dict):
        result_dict = dict()
        result_dict['M'] = self.__M
        result_dict['R'] = self.__R
        result_dict['K'] = self.__K
        result_dict['O'] = self.__O
        result_dict['alpha'] = self.__alpha
        result_dict['beta'] = self.__beta
        result_dict['gamma'] = self.__gamma
        result_dict['iteration'] = self.__iteration

        now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir(os.path.join(folder_path, 'training_result_'+now))
        folder_path = os.path.join(folder_path, 'training_result_'+now)
        with open(os.path.join(folder_path, 'result_parameter.csv'.format(now)), "w",
                  encoding='utf-8-sig', newline='') as f:
            basic_data = [['parameter', 'value']]
            for key in result_dict:
                basic_data.append([key, result_dict[key]])
            csv.writer(f).writerows(basic_data)

        data_to_write = list()
        head = list()
        head.append('')
        for i in range(self.__K):
            head.append('state {}'.format(i + 1))
        data_to_write.append(head)
        for i, line in enumerate(self.__n_z):
            line_ = list()
            line_.append('state {}'.format(i + 1))
            for item in line:
                line_.append(item)
            data_to_write.append(line_)
        with open(os.path.join(folder_path, 'n_z.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        data_to_write = list()
        head = list()
        head.append('')
        for i in range(self.__R):
            head.append('risk tier {}'.format(i+1))
        data_to_write.append(head)
        for i, line in enumerate(self.__n_r):
            line_ = list()
            line_.append('state {}'.format(i+1))
            for item in line:
                line_.append(item)
            data_to_write.append(line_)
        with open(os.path.join(folder_path, 'n_r.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        n_o = self.__n_o
        with open(os.path.join(folder_path, 'n_o.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            n_o_tensor = list()
            for i in range(len(n_o)):
                n_o_tensor.append(['state {}'.format(i+1)])
                caption = list()
                caption.append('')
                for j in range(len(index_name_dict)):
                    caption.append(index_name_dict[j])
                n_o_tensor.append(caption)
                for j in range(len(n_o[i])):
                    single_line = ['risk tier {}'.format(j+1)]
                    for k in range(len(n_o[i][j])):
                        single_line.append(n_o[i][j][k])
                    n_o_tensor.append(single_line)
            csv.writer(f).writerows(n_o_tensor)

        data_to_write = list()
        head = list()
        head.append('')
        for i in range(self.__K):
            head.append('state {}'.format(i + 1))
        data_to_write.append(head)
        for i, line in enumerate(self.__Theta):
            line_ = list()
            line_.append('state {}'.format(i + 1))
            for item in line:
                line_.append(item)
            data_to_write.append(line_)
        with open(os.path.join(folder_path, 'theta.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        data_to_write = list()
        head = list()
        head.append('')
        for i in range(self.__R):
            head.append('risk tier {}'.format(i + 1))
        data_to_write.append(head)
        for i, line in enumerate(self.__Pi):
            line_ = list()
            line_.append('state {}'.format(i + 1))
            for item in line:
                line_.append(item)
            data_to_write.append(line_)
        with open(os.path.join(folder_path, 'pi.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        with open(os.path.join(folder_path, 'phi.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            phi_tensor = list()
            for i in range(len(self.__Phi)):
                phi_tensor.append(['state {}'.format(i + 1)])
                caption = list()
                caption.append('')
                for j in range(len(index_name_dict)):
                    caption.append(index_name_dict[j])
                phi_tensor.append(caption)
                for j in range(len(self.__Phi[i])):
                    single_line = ['risk tier {}'.format(j + 1)]
                    for k in range(len(self.__Phi[i][j])):
                        single_line.append(self.__Phi[i][j][k])
                    phi_tensor.append(single_line)
                phi_tensor.append([])
                phi_tensor.append([])
            csv.writer(f).writerows(phi_tensor)

        with open(os.path.join(folder_path, 'integrate_phi.csv'.format(now)), "w", encoding='utf-8-sig',
                  newline='') as f:
            integrate_phi_mat = list()
            caption = list()
            caption.append('state')
            for i in range(len(index_name_dict)):
                caption.append(index_name_dict[i])
            integrate_phi_mat.append(caption)

            for i in range(len(self.__Phi)):
                phi_state = self.__Phi[i]
                phi_integrate_state = np.sum(phi_state, axis=0)
                line = list()
                line.append('state {}'.format(i+1))
                for j in range(len(phi_integrate_state)):
                    line.append(phi_integrate_state[j])
                integrate_phi_mat.append(line)
            csv.writer(f).writerows(integrate_phi_mat)

        with open(os.path.join(folder_path, 'hidden_state_assignment.csv'.format(now)),
                  "w", encoding='utf-8-sig', newline='') as f:
            hidden = self.__hidden_state_assignment
            hidden_matrix = list()
            hidden_matrix.append(['patient_id', 'visit_index', 'subtype', 'risk_tier'])
            for i in range(len(hidden)):
                patient_id = index_patient_dict[i]
                for j in range(len(hidden[i])):
                    hidden_matrix.append([patient_id, j, hidden[i][j][0], hidden[i][j][1]])
            csv.writer(f).writerows(hidden_matrix)


def synthetic_data_generator(num_trajectory, max_visit, min_visit, max_observation_length, min_observation_length,
                             unique_observation):
    if max_visit < min_visit or max_observation_length < min_observation_length \
     or max_observation_length > unique_observation:
        raise ValueError('')

    data = list()
    for i in range(num_trajectory):
        seq_length = random.randint(min_visit, max_visit)
        data.append(list())
        for j in range(seq_length):
            target_list = data[-1]
            target_list.append(list())
            observation_length = random.randint(min_observation_length, max_observation_length)

            valid_observation = [i for i in range(unique_observation)]
            for k in range(observation_length):
                index = random.randint(0, len(valid_observation)-1)
                target_list[-1].append(valid_observation.pop(index))
    return data


def save_risk_tier_assignment(estimated_list, index_patient_dict, folder_path):
    data_to_write = list()
    head = ['patient_id', 'visit_index']
    for i, _ in enumerate(estimated_list[0]):
        head.append('risk tier {}'.format(i+1))
    data_to_write.append(head)

    for i, trajectory_list in enumerate(estimated_list):
        patient_id = index_patient_dict[i]
        single_visit_risk = [patient_id]
        for k in trajectory_list:
            single_visit_risk.append(k)
        data_to_write.append(single_visit_risk)

    with open(os.path.join(folder_path, 'risk_tier_inference.csv'), "w",
              encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def save_disease_state_assignment(estimated_list, index_patient_dict, folder_path):
    data_to_write = list()
    data_to_write.append(['patient_id', 'visit_index', 'hidden_state'])
    for i, trajectory_list in enumerate(estimated_list):
        patient_id = index_patient_dict[i]
        for j, hidden_state in enumerate(trajectory_list):
            data_to_write.append([patient_id, j, hidden_state])

    with open(os.path.join(folder_path, 'state_assignment.csv'), "w",
              encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def unit_test():
    file_path = os.path.abspath('../../resource/二值化后的长期纵向数据.csv')
    document, index_name_dict, index_patient_dict = data_reader.data_reader(file_path)

    risk_tier = 3
    patient_subtype = 5
    alpha = 0.1
    beta = 0.1
    gamma = 0.1
    disease_progression_model = DiseaseProgressionModel(document, alpha=alpha, beta=beta, gamma=gamma,
                                                        num_risk_tier=risk_tier, num_subtype=patient_subtype)

    disease_progression_model.optimization(10)
    estimated_risk_list = disease_progression_model.risk_tier_inference(document)
    estimated_state_list = disease_progression_model.disease_state_assignment(document)

    model_save_path = os.path.abspath('../../resource/result/DPM')
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    os.mkdir(os.path.join(model_save_path, 'inference_result_' + now))
    inference_path = os.path.join(model_save_path, 'inference_result_' + now)

    disease_progression_model.save_result(model_save_path, index_name_dict, index_patient_dict)
    save_disease_state_assignment(estimated_state_list, index_patient_dict, inference_path)
    save_risk_tier_assignment(estimated_risk_list, index_patient_dict, inference_path)


if __name__ == '__main__':
    unit_test()
