# encoding=utf-8
import numpy as np
import random
import os
import scipy as sp
import datetime
import csv
from math import log, exp
import data_reader


class DiseaseProgressionModel(object):
    def __init__(self, data, num_risk_tier, num_subtype, alpha, beta, gamma, dependence_constraint=2,
                 init_state_candidate=2, parameter_init_folder=None):
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
        self.__loss_list = [['iteration count', 'log likelihood']]
        self.__dependence_constraint = dependence_constraint
        self.__init_state_candidate = init_state_candidate
        self.__parameter_init_folder = parameter_init_folder
        if init_state_candidate > num_subtype:
            raise ValueError('')

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

        self.__Theta = np.zeros([self.__K + 1, self.__K])
        self.__Pi = np.zeros([self.__K, self.__R])
        self.__Phi = np.zeros([self.__K, self.__R, self.__O])
        if self.__parameter_init_folder is not None:
            phi_path = os.path.join(self.__parameter_init_folder, 'phi.csv')
            theta_path = os.path.join(self.__parameter_init_folder, 'theta.csv')
            pi_path = os.path.join(self.__parameter_init_folder, 'pi.csv')
            with open(phi_path, 'r', encoding='utf-8-sig', newline='') as f:
                csv_reader = csv.reader(f)
                for i, line in enumerate(csv_reader):
                    if i == 0 or i % 6 == 0 or i % 6 == 1 or i % 6 == 4 or i % 6 == 5:
                        continue
                    for j, item in enumerate(line):
                        if j == 0:
                            continue
                        self.__Phi[i//6, i % 6-2, j-1] = float(item)
            with open(theta_path, 'r', encoding='utf-8-sig', newline='') as f:
                csv_reader = csv.reader(f)
                for i, line in enumerate(csv_reader):
                    if i == 0:
                        continue
                    for j, item in enumerate(line):
                        if j == 0:
                            continue
                        self.__Theta[i-1, j-1] = float(item)
            with open(pi_path, 'r', encoding='utf-8-sig', newline='') as f:
                csv_reader = csv.reader(f)
                for i, line in enumerate(csv_reader):
                    if i == 0:
                        continue
                    for j, item in enumerate(line):
                        if j == 0:
                            continue
                        self.__Pi[i-1, j-1] = float(item)
        else:
            # initialize parameter
            # note all parameters follows categorical distribution, i.e., the sum of each row (2D case) should be 1
            # note the markov chain needs a initial state, so the size of state space is K+1, we use the state
            # whose idx is K to denote the initial state
            # we define the initial state is K

            # normalization
            for i in range(len(self.__Pi)):
                sample = np.random.dirichlet(np.full(self.__R, self.__gamma))
                for j in range(len(self.__Pi[i])):
                    self.__Pi[i, j] = sample[j]
            for i in range(len(self.__Phi)):
                for j in range(len(self.__Phi[i])):
                    sample = np.random.dirichlet(self.__beta)
                    for k in range(len(self.__Phi[i, j])):
                        self.__Phi[i, j, k] = sample[k]

            # initialize Theta with constraint
            # Theta is a upper triangular matrix
            constraint = self.__dependence_constraint
            # init state
            init_trans_distribution = np.random.dirichlet(np.full(self.__init_state_candidate, self.__alpha))
            for i in range(self.__init_state_candidate):
                self.__Theta[self.__K, i] = init_trans_distribution[i]
            # transition state
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
        self.__hidden_state_assignment = list()
        # assignment
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            self.__hidden_state_assignment.append(list())
            for j in range(len(trajectory)):
                if j == 0:
                    current_state = random.randint(0, self.__init_state_candidate-1)
                else:
                    # the latter state can't be bigger than the previous one
                    latter_state_candidate = self.__hidden_state_assignment[-1][-1][0] + self.__dependence_constraint
                    current_state = self.__hidden_state_assignment[-1][-1][0]
                    if latter_state_candidate >= self.__K:
                        latter_state_candidate = self.__K - 1
                    current_state = random.randint(current_state, latter_state_candidate)
                risk_tier = random.randint(0, self.__R-1)
                self.__hidden_state_assignment[-1].append([current_state, risk_tier])

        # the index of initial state is K
        self.__n_z = np.zeros([self.__K+1, self.__K])
        self.__n_r = np.zeros([self.__K, self.__R])
        self.__n_o = np.zeros([self.__K, self.__R, self.__O])
        # updating count
        for i in range(len(self.__hidden_state_assignment)):
            trajectory = self.__hidden_state_assignment[i]
            for j in range(len(trajectory)):
                # updating n_z
                current_state = self.__hidden_state_assignment[i][j][0]
                if j == 0:
                    previous_state = self.__K
                else:
                    previous_state = self.__hidden_state_assignment[i][j-1][0]
                self.__n_z[previous_state, current_state] += 1

                # updating n_r
                risk_tier = self.__hidden_state_assignment[i][j][1]
                self.__n_r[current_state, risk_tier] += 1

                # updating n_o
                obs_sequence = self.__data[i][j]
                for k in range(len(obs_sequence)):
                    observation = obs_sequence[k]
                    self.__n_o[current_state, risk_tier, observation] += 1
        print('initialize procedure accomplished')

    def optimization(self, iteration_num, update_interval=10):
        start_time = datetime.datetime.now()

        likelihood = self.log_likelihood()
        print('initial log likelihood {}'. format(likelihood))
        self.__loss_list.append([0, likelihood])

        for i in range(1, iteration_num+1):
            self.__iteration = iteration_num
            self.__inference()
            if i % update_interval == 0 or i == iteration_num:
                self.__parameter_updating()
                likelihood = self.log_likelihood()
                end_time = datetime.datetime.now()
                print('iteration {} accomplished, current log-likelihood {}, time cost {} s'.
                      format(i, likelihood, (end_time-start_time).seconds))
                start_time = datetime.datetime.now()
                self.__loss_list.append([i, likelihood])

    def __inference(self):
        # updating z
        for i in range(len(self.__data)):
            for j in range(len(self.__data[i])):
                self.__sampling_state(i, j)
        # updating r
        for i in range(len(self.__data)):
            for j in range(len(self.__data[i])):
                self.__sampling_risk_tier(i, j)

    def __sampling_state(self, i, j):
        # Step 1: decrease count
        current_state, risk_tier = self.__hidden_state_assignment[i][j]
        if j == 0:
            previous_state = self.__K
            next_state = self.__hidden_state_assignment[i][j+1][0]
        elif j == len(self.__hidden_state_assignment[i])-1:
            previous_state = self.__hidden_state_assignment[i][j-1][0]
            # in fact, the final state doesn't have descendant state
            next_state = None
        else:
            previous_state = self.__hidden_state_assignment[i][j-1][0]
            next_state = self.__hidden_state_assignment[i][j+1][0]
        # minus hidden state count
        self.__n_z[previous_state][current_state] -= 1
        if next_state is not None:
            self.__n_z[current_state][next_state] -= 1
        self.__n_r[current_state][risk_tier] -= 1
        observation_list = self.__data[i][j]
        for observation in observation_list:
            self.__n_o[current_state][risk_tier][observation] -= 1

        # Step 2: calculate probability with constraint, follow Eq. 21
        # first visit check
        if previous_state == self.__K:
            candidate_state_count = self.__init_state_candidate
        else:
            candidate_state_count = self.__K - previous_state
            if candidate_state_count > self.__dependence_constraint + 1:
                candidate_state_count = self.__dependence_constraint + 1

        # the first visit can be arbitrary state
        if j == 0:
            start_state = 0
            end_state = candidate_state_count
        else:
            start_state = previous_state
            end_state = previous_state + candidate_state_count

        # For part 1, follow Eq. 16, 17
        part_1 = np.zeros([candidate_state_count])
        # last visit, using the Eq. 17 to calculate the part 1
        if j == len(self.__N[i])-1:
            for k in range(start_state, end_state):
                numerator = self.__alpha + self.__n_z[previous_state][k]
                denominator = self.__alpha * candidate_state_count + np.sum(self.__n_z[previous_state])
                part_1[k-start_state] = numerator/denominator
        # not the last visit, using the Eq. 16 to calculate the part 1
        else:
            next_state = self.__hidden_state_assignment[i][j + 1][0]
            for k in range(start_state, end_state):
                numerator_1 = self.__alpha + self.__n_z[previous_state][k]
                numerator_2 = self.__alpha + self.__n_z[k][next_state]
                if k == next_state and k == previous_state:
                    numerator_2 += 1
                denominator_1 = self.__alpha * candidate_state_count + np.sum(self.__n_z[previous_state])
                denominator_2 = self.__alpha * candidate_state_count + np.sum(self.__n_z[k])
                if previous_state == k:
                    denominator_2 += 1
                part_1[k-start_state] = (numerator_1*numerator_2) / (denominator_1*denominator_2)

        # for part 2, follow equation 18
        part_2 = np.zeros([candidate_state_count])
        for k in range(start_state, end_state):
            numerator = self.__gamma + self.__n_r[k][risk_tier]
            denominator = self.__gamma * self.__R + np.sum(self.__n_r[k])
            part_2[k-start_state] = numerator / denominator

        # for part 3, follow equation 20
        part_3 = np.zeros([candidate_state_count])
        for k in range(start_state, end_state):
            part_3_obs = 1
            denominator = np.sum(self.__beta) + np.sum(self.__n_o[k, risk_tier])
            for obs in self.__data[i][j]:
                numerator = self.__beta[obs] + self.__n_o[k, risk_tier, obs]
                part_3_obs = part_3_obs * numerator / denominator
            part_3[k-start_state] = part_3_obs

        # normalization
        unnormalized_probability = part_1 * part_2 * part_3
        # if we meet underflow problem, the next line will throw an Exception
        normalized_probability = unnormalized_probability / np.sum(unnormalized_probability)
        cumulative_probability = np.zeros(normalized_probability.shape)
        for a in range(len(cumulative_probability)):
            for b in range(a+1):
                cumulative_probability[a] += normalized_probability[b]

        # Step 3: sampling new assignment
        random_number = random.uniform(0, 1)
        current_state_new_sample = None
        for a in range(0, len(cumulative_probability)):
            if random_number <= cumulative_probability[a]:
                current_state_new_sample = a
                break

        if j != 0:
            current_state_new_sample = current_state_new_sample + self.__hidden_state_assignment[i][j-1][0]

        # Step 4: recover count
        self.__n_z[previous_state][current_state_new_sample] += 1
        if next_state is not None:
            self.__n_z[current_state_new_sample][next_state] += 1
        self.__n_r[current_state_new_sample][risk_tier] += 1
        for observation in observation_list:
            self.__n_o[current_state_new_sample][risk_tier][observation] += 1
        self.__hidden_state_assignment[i][j][0] = current_state_new_sample

    def __sampling_risk_tier(self, i, j):
        # Step 1: decrease count
        current_state, current_risk_tier = self.__hidden_state_assignment[i][j]
        self.__n_r[current_state][current_risk_tier] -= 1
        observation_list = self.__data[i][j]
        for observation in observation_list:
            self.__n_o[current_state][current_risk_tier][observation] -= 1

        # Step 2: calculate probability
        # follow Equation 23.
        # For Part 1
        part_1_numerator = self.__gamma + self.__n_r[current_state]
        part_1_denominator = np.sum(self.__n_r[current_state]) + self.__gamma * self.__R
        part_1 = part_1_numerator / part_1_denominator
        # For Part 2
        part_2 = 1
        part_2_denominator = np.sum(self.__beta) + np.sum(self.__n_o[current_state, :, :], axis=1)
        for observation in self.__data[i][j]:
            part_2_numerator = self.__beta[observation] + self.__n_o[current_state, :, observation]
            part_2 = part_2 * part_2_numerator / part_2_denominator

        unnormalized_probability = part_1 * part_2
        normalized_probability = unnormalized_probability / np.sum(unnormalized_probability)
        cumulative_probability = np.zeros(normalized_probability.shape)
        for k in range(len(cumulative_probability)):
            for l in range(k+1):
                cumulative_probability[k] += normalized_probability[l]

        # Step 3: sampling new assignment
        random_number = random.uniform(0, 1)
        new_risk_tier = None
        for k in range(0, len(cumulative_probability)):
            if random_number <= cumulative_probability[k]:
                new_risk_tier = k
                break

        # Step 4: recover count
        self.__n_r[current_state][new_risk_tier] += 1
        for observation in self.__data[i][j]:
            self.__n_o[current_state][new_risk_tier][observation] += 1
        self.__hidden_state_assignment[i][j][1] = new_risk_tier

    def __parameter_updating(self):
        # updating Theta with constraint
        for i in range(self.__K+1):
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
        for i in range(self.__K):
            for j in range(self.__R):
                self.__Pi[i][j] = (self.__n_r[i][j] + self.__gamma) / np.sum(self.__n_r[i] + self.__gamma)
                if self.__Pi[i][j] < 0:
                    print('ERROR')
        # updating Phi
        for i in range(self.__K):
            for j in range(self.__R):
                for k in range(self.__O):
                    self.__Phi[i][j][k] = \
                        (self.__n_o[i][j][k] + self.__beta[k]) / np.sum(self.__n_o[i][j] + self.__beta)
                    if self.__Phi[i][j][k] < 0:
                        print('ERROR')

    def log_likelihood(self, test_data=None):
        """
        :param test_data: if test data is not None, calculate the likelihood of test data
                          if test data is None, calculate the likelihood of training data
        :return:
        """

        # calculate log s
        # Follow Algorithm 1
        def calculate_part_1(cache):
            log_s = 0
            # note in this case, we don't need do care about the initial state, i.e., l = self.__K
            for k in range(self.__K):
                current_sum = cache[k]
                if k == 0:
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
        # follow Eq. 28
        for patient_index in range(len(forward_cache)):
            likelihood += calculate_part_1(forward_cache[patient_index])
        return likelihood

    # calculate log p given j, k
    def __calculate_part_3(self, trajectory, visit_id, state):
        # Follow Algorithm 3
        j, k = visit_id, state
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
    def __forward_calculate_part_2(self, trajectory, cache, visit_id, state):
        # Follow Eq. 26, 27, Algorithm 2
        j, k = visit_id, state
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
                # It is impossible for some cases to accomplish transition like init state (e.g. state 5) -> state 3
                # to ensure numerical stability, we will assign a very small value for those impossible transition
                if self.__Theta[self.__K, k] == 0:
                    current_sum = -10000 + log_p
                else:
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

        if log_q >= -0.1:
            raise ValueError('')
        return log_q

    def __backward_calculate_part_2(self, trajectory, cache, visit_id, state):
        j, k = visit_id, state
        log_q = 0

        if j == len(trajectory)-1:
            return 1

        # backward procedure path
        for l_ in range(self.__K):
            # constraint
            if self.__Theta[k, l_] == 0:
                continue

            log_p = self.__calculate_part_3(trajectory, j+1, l_)

            current_sum = cache[-1][l_] + log(self.__Theta[k, l_]) + log_p

            if log_q == 0:
                log_q = current_sum
            else:
                if log_q > current_sum:
                    log_q = log_q + log(1 + exp(current_sum - log_q))
                else:
                    log_q = current_sum + log(1 + exp(log_q - current_sum))

            if log_q > 0:
                raise ValueError('')
        return log_q

    def __forward_procedure(self, trajectory, terminate_idx):
        # Follow Eq. 26, 27
        forward_cache = list()
        for visit_index in range(terminate_idx):
            a_ij = list()
            for patient_state in range(self.__K):
                a_ijk = self.__forward_calculate_part_2(trajectory, forward_cache, visit_index, patient_state)
                a_ij.append(a_ijk)
            forward_cache.append(a_ij)
        return forward_cache[-1]

    def __backward_procedure(self, trajectory, start_idx):
        backward_cache = list()
        for visit_index in range(start_idx, len(trajectory)).__reversed__():
            a_ij = list()
            for patient_state in range(self.__K):
                a_ijk = self.__backward_calculate_part_2(trajectory, backward_cache, visit_index, patient_state)
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

            for j, single_visit in enumerate(single_trajectory_list):
                # to avoid the underflow problem, we use the log likelihood

                # calculate the observation probability
                observation_log_prob_mat = np.zeros([num_state, num_risk])
                for k in range(num_state):
                    for r in range(num_risk):
                        observation_log_prob = 0
                        for observation in single_trajectory_list[j]:
                            observation_log_prob += log(phi[k, r, observation])
                        observation_log_prob_mat[k, r] = observation_log_prob

                # init step
                if j == 0:
                    prev_state = num_state
                    for k in range(num_state):
                        log_prob = 0
                        for r in range(num_risk):
                            if r == 0:
                                log_prob = log(pi[k, r]) + observation_log_prob_mat[k, r]
                            else:
                                temp = log(pi[k, r]) + observation_log_prob_mat[k, r]
                                if log_prob > temp:
                                    log_prob = log_prob + log(1 + exp(temp - log_prob))
                                else:
                                    log_prob = temp + log(1 + exp(log_prob - temp))

                        if theta[prev_state, k] == 0:
                            log_prob += -300
                        else:
                            log_prob += log(theta[prev_state, k])
                        delta[j, k] = log_prob
                        psi[j, k] = -10000
                    continue

                # recursion step
                for k in range(num_state):
                    max_prob = -float('inf')
                    for prev_state in range(num_state):
                        log_prob = 0

                        # To avoid the numerical unstable problem when we constraint the state transition direction
                        if theta[prev_state, k] == 0:
                            log_transition_prob = -100000
                        else:
                            log_transition_prob = log(theta[prev_state, k])

                        for r in range(num_risk):
                            if r == 0:
                                log_prob = delta[j-1, prev_state] + log_transition_prob + log(pi[k, r]) + \
                                           observation_log_prob_mat[k, r]
                            else:
                                temp = delta[j-1, prev_state] + log_transition_prob + log(pi[k, r]) + \
                                       observation_log_prob_mat[k, r]
                                if log_prob > temp:
                                    log_prob = log_prob + log(1 + exp(temp - log_prob))
                                else:
                                    log_prob = temp + log(1 + exp(log_prob - temp))
                        if log_prob > max_prob:
                            max_prob = log_prob
                            delta[j, k] = log_prob
                            psi[j, k] = prev_state

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
            # hidden state inference procedure, Follow Eq.34
            idx = len(trajectory)+visit_idx

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
                            denominator_obs *= phi[hidden_state, r, trajectory[idx][n_]]
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
        result_dict['dependence_constraint'] = self.__dependence_constraint

        with open(os.path.join(folder_path, 'result_parameter.csv'), "w",
                  encoding='utf-8-sig', newline='') as f:
            basic_data = [['parameter', 'value']]
            for key in result_dict:
                basic_data.append([key, result_dict[key]])
            csv.writer(f).writerows(basic_data)

        # state transition count
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
        with open(os.path.join(folder_path, 'n_z.csv'), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # state risk tier count
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
        with open(os.path.join(folder_path, 'n_r.csv'), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # state risk tier observation count
        n_o = self.__n_o
        with open(os.path.join(folder_path, 'n_o.csv'), "w", encoding='utf-8-sig', newline='') as f:
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
                n_o_tensor.append([])
                n_o_tensor.append([])
            csv.writer(f).writerows(n_o_tensor)

        # Theta
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
        with open(os.path.join(folder_path, 'theta.csv'), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # Pi
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
        with open(os.path.join(folder_path, 'pi.csv'), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # Phi
        with open(os.path.join(folder_path, 'phi.csv'), "w", encoding='utf-8-sig', newline='') as f:
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

        # Phi
        with open(os.path.join(folder_path, 'integrate_phi.csv'), "w", encoding='utf-8-sig',
                  newline='') as f:
            integrate_phi_mat = list()
            caption = list()
            caption.append('')
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

        with open(os.path.join(folder_path, 'hidden_state_assignment.csv'),
                  "w", encoding='utf-8-sig', newline='') as f:
            hidden = self.__hidden_state_assignment
            hidden_matrix = list()
            hidden_matrix.append(['patient_id', 'visit_index', 'state', 'risk_tier'])
            for i in range(len(hidden)):
                patient_id = index_patient_dict[i]
                for j in range(len(hidden[i])):
                    hidden_matrix.append([patient_id, j, hidden[i][j][0], hidden[i][j][1]])
            csv.writer(f).writerows(hidden_matrix)

        with open(os.path.join(folder_path, 'likelihood_change.csv'),
                  "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__loss_list)


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
    head = ['patient_id']
    for i, _ in enumerate(estimated_list[0]):
        head.append('risk tier {}'.format(i+1))
    data_to_write.append(head)

    for i, trajectory_list in enumerate(estimated_list):
        patient_id = index_patient_dict[i]
        single_visit_risk = [patient_id]
        for k in trajectory_list:
            single_visit_risk.append(k)
        data_to_write.append(single_visit_risk)

    with open(os.path.join(folder_path, 'risk_tier_inference_of_last_visit.csv'), "w",
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

    obs_num = 107
    risk_tier = 2
    patient_subtype = 4
    alpha = 1
    beta = np.full(obs_num, 0.1)
    gamma = 0.1
    disease_progression_model = DiseaseProgressionModel(document, alpha=alpha, beta=beta, gamma=gamma,
                                                        num_risk_tier=risk_tier, num_subtype=patient_subtype)

    disease_progression_model.optimization(1, update_interval=1)
    estimated_risk_list = disease_progression_model.risk_tier_inference(document)
    estimated_state_list = disease_progression_model.disease_state_assignment(document)

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_save_path = os.path.abspath('../../resource/result/DPM/{}'.format(now))
    os.mkdir(model_save_path)
    os.mkdir(os.path.join(model_save_path, 'inference_result'))
    inference_path = os.path.join(model_save_path, 'inference_result')

    disease_progression_model.save_result(model_save_path, index_name_dict, index_patient_dict)
    save_disease_state_assignment(estimated_state_list, index_patient_dict, inference_path)
    save_risk_tier_assignment(estimated_risk_list, index_patient_dict, inference_path)


if __name__ == '__main__':
    unit_test()
