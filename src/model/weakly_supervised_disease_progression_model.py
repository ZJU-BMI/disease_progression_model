# encoding=utf-8
import numpy as np
import random
import os
import datetime
import csv
from math import log, exp


class WeaklySupervisedDiseaseProgressionModel(object):
    def __init__(self, data, num_risk_tier, num_subtype, alpha, beta, gamma):
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

        # initialize all variables
        self.initialization()

    def initialization(self):
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
        self.__Theta = np.random.uniform(0, 1, [self.__K+1, self.__K+1])
        self.__Pi = np.random.uniform(0, 1, [self.__K+1, self.__R])
        self.__Phi = np.random.uniform(0, 1, [self.__K+1, self.__R, self.__O])
        # normalization
        for i in range(len(self.__Theta)):
            row_sum = np.sum(self.__Theta[i])
            self.__Theta[i] = self.__Theta[i]/row_sum
        for i in range(len(self.__Pi)):
            row_sum = np.sum(self.__Pi[i])
            self.__Pi[i] = self.__Pi[i]/row_sum
        for i in range(len(self.__Phi)):
            for j in range(len(self.__Phi[i])):
                row_sum = np.sum(self.__Phi[i][j])
                self.__Phi[i][j] = self.__Phi[i][j]/row_sum

        # initialize topic assignment
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
                current_subtype = random.randint(0, self.__K-1)
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
            self.inference()
            self.parameter_updating()
            likelihood = self.log_likelihood()
            end_time = datetime.datetime.now()
            print('iteration {} accomplished, current log-likelihood {}, time cost {}'.
                  format(i+1, likelihood, end_time-start_time))

    def inference(self):
        # updating z
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            for j in range(len(trajectory)):
                self.sampling_subtype(i, j)
        # updating r
        for i in range(len(self.__data)):
            trajectory = self.__data[i]
            for j in range(len(trajectory)):
                self.sampling_risk_tier(i, j)

    def sampling_subtype(self, i, j):
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

        # Step 2: calculate probability
        # for part 1
        if next_subtype == no_next_indicator:
            # follow Equation 17
            part_1 = (self.__alpha + self.__n_z[previous_subtype]) / np.sum(self.__alpha+self.__n_z[previous_subtype])
        else:
            # follow Equation 16
            part_1_1 = (self.__alpha + self.__n_z[previous_subtype]) / np.sum(self.__alpha+self.__n_z[previous_subtype])
            part_1_2_numerator = self.__alpha[next_subtype] + self.__n_z[:, next_subtype]
            if next_subtype == previous_subtype:
                part_1_2_numerator[next_subtype] += 1
            part_1_2_denominator = np.sum(self.__alpha) + np.sum(self.__n_z, axis=1)
            part_1_2_denominator[previous_subtype] += 1
            part_1 = part_1_1 * part_1_2_numerator / part_1_2_denominator

        # for part 2
        part_2_numerator = self.__gamma[risk_tier] + self.__n_r[:, risk_tier]
        part_2_denominator = np.sum(self.__gamma) + np.sum(self.__n_r, axis=1)
        part_2 = part_2_numerator / part_2_denominator
        # for part 3
        part_3 = 1
        part_3_denominator = np.sum(self.__beta) + np.sum(self.__n_o[:, risk_tier, :], axis=1)
        for observation in observation_list:
            part_3_numerator = self.__beta[observation] + self.__n_o[:, risk_tier, observation]
            part_3 = part_3 * part_3_numerator / part_3_denominator

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

        # Step 4: recover count
        self.__n_z[previous_subtype][new_subtype_sample] += 1
        if next_subtype != no_next_indicator:
            self.__n_z[new_subtype_sample][next_subtype] += 1
        self.__n_r[new_subtype_sample][risk_tier] += 1
        for observation in observation_list:
            self.__n_o[new_subtype_sample][risk_tier][observation] += 1
        self.__hidden_state_assignment[i][j][0] = new_subtype_sample

    def sampling_risk_tier(self, i, j):
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
        part_2_denominator = np.sum(self.__beta) + np.sum(self.__n_o[current_subtype, :, :], axis=1)
        for observation in self.__data[i][j]:
            part_2_numerator = self.__beta[observation] + self.__n_o[current_subtype, :, observation]
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

    def parameter_updating(self):
        # updating Theta
        for i in range(self.__K + 1):
            for j in range(self.__K + 1):
                self.__Theta[i][j] = (self.__n_z[i][j] + self.__alpha[j]) / np.sum(self.__n_z[i] + self.__alpha)
                if self.__Theta[i][j] < 0:
                    print('ERROR')
        # updating Pi
        for i in range(self.__K + 1):
            for j in range(self.__R):
                self.__Pi[i][j] = (self.__n_r[i][j] + self.__gamma[j]) / np.sum(self.__n_r[i] + self.__gamma)
                if self.__Pi[i][j] < 0:
                    print('ERROR')
        # updating Phi
        for i in range(self.__K + 1):
            for j in range(self.__R):
                for k in range(self.__O):
                    self.__Phi[i][j][k] = \
                        (self.__n_o[i][j][k] + self.__beta[k]) / np.sum(self.__n_o[i][j] + self.__beta)
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
        if test_data is None:
            data = self.__data
        else:
            data = test_data

        # calculate log p given i, j, k
        def calculate_part_3(pat_index, visit_id, subtype):
            i, j, k = pat_index, visit_id, subtype
            log_p = 0
            for m in range(self.__R):
                current_sum = log(self.__Pi[k, m])
                for n in range(self.__N[i][j]):
                    current_sum += log(self.__Phi[k, m, data[i][j][n]])

                if m == 0:
                    log_p = current_sum
                else:
                    if log_p > current_sum:
                        log_p = log_p + log(1 + exp(current_sum - log_p))
                    else:
                        log_p = current_sum + log(1 + exp(log_p - current_sum))
            return log_p

        # calculate log q
        def calculate_part_2(cache, pat_index, visit_id, subtype):
            i, j, k = pat_index, visit_id, subtype
            log_q = 0

            # note in this case, we don't need do care about the initial state, i.e., l = self.__K
            for l_ in range(self.__K):
                log_p = calculate_part_3(i, j, k)
                if j == 0:
                    current_sum = log(self.__Theta[l_, k]) + log_p
                else:
                    current_sum = cache[i][-1][l_] + log(self.__Theta[l_, k]) + log_p

                if l_ == 0:
                    log_q = current_sum
                else:
                    if log_q > current_sum:
                        log_q = log_q + log(1 + exp(current_sum - log_q))
                    else:
                        log_q = current_sum + log(1 + exp(log_q - current_sum))
            return log_q

        # calculate log s
        def calculate_part_1(cache, pat_index):
            i = pat_index
            log_s = 0
            # note in this case, we don't need do care about the initial state, i.e., l = self.__K
            for k in range(self.__K):
                current_sum = cache[i][-1][k]
                if k == 1:
                    log_s = current_sum
                else:
                    if log_s > current_sum:
                        log_s = log_s + log(1 + exp(current_sum - log_s))
                    else:
                        log_s = current_sum + log(1 + exp(log_s - current_sum))
            return log_s

        # cache initialization
        forward_cache = list()
        for patient_index in range(self.__M):
            forward_cache.append(list())
            for visit_index in range(self.__T[patient_index]):
                a_ij = list()
                for patient_subtype in range(self.__K):
                    a_ijk = calculate_part_2(forward_cache, patient_index, visit_index, patient_subtype)
                    a_ij.append(a_ijk)
                forward_cache[-1].append(a_ij)

        likelihood = 0
        for patient_index in range(self.__M):
            likelihood += calculate_part_1(forward_cache, patient_index)
        return likelihood

    def save_result(self, folder_path):
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
        result_dict['n_r'] = self.__n_r
        with open(os.path.join(folder_path, 'n_r.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__n_r)

        n_o = self.__n_o
        result_dict['n_o'] = self.__n_o
        with open(os.path.join(folder_path, 'n_o.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            n_o_matrix = list()
            for i in range(len(n_o)):
                n_o_matrix.append(['i', i])
                for j in range(len(n_o[i])):
                    n_o_matrix.append(n_o[i][j])
            csv.writer(f).writerows(n_o_matrix)

        result_dict['Theta'] = self.__Theta
        with open(os.path.join(folder_path, 'theta.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__Theta)
        result_dict['Pi'] = self.__Pi
        with open(os.path.join(folder_path, 'pi.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(self.__Pi)
        phi = self.__Phi
        result_dict['phi'] = self.__Phi
        with open(os.path.join(folder_path, 'phi.csv'.format(now)), "w", encoding='utf-8-sig', newline='') as f:
            phi_matrix = list()
            for i in range(len(phi)):
                phi_matrix.append(['i', i])
                for j in range(len(phi[i])):
                    n_o_matrix.append(phi[i][j])
            csv.writer(f).writerows(phi_matrix)

        with open(os.path.join(folder_path, 'hidden_state_assignment.csv'.format(now)),
                  "w", encoding='utf-8-sig', newline='') as f:
            hidden = self.__hidden_state_assignment
            hidden_matrix = list()
            hidden_matrix.append(['patient_index', 'visit_index', 'subtype', 'risk_tier'])
            for i in range(len(hidden)):
                for j in range(len(hidden[i])):
                    hidden_matrix.append([i, j, hidden[i][j][0], hidden[i][j][1]])
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


def unit_test():
    num_seq = 300
    max_visit = 10
    min_visit = 4
    max_observation = 80
    min_observation = 50
    unique_observation = 400
    document = synthetic_data_generator(num_seq, max_visit, min_visit, max_observation, min_observation,
                                        unique_observation)

    risk_tier = 3
    patient_subtype = 4
    alpha = [0.1 for _ in range(patient_subtype + 1)]
    beta = [0.1 for _ in range(unique_observation)]
    gamma = [0.1 for _ in range(risk_tier)]
    disease_progression_model = WeaklySupervisedDiseaseProgressionModel(document, alpha=alpha, beta=beta, gamma=gamma,
                                                                        num_risk_tier=risk_tier,
                                                                        num_subtype=patient_subtype)
    disease_progression_model.optimization(10)
    disease_progression_model.save_result(os.path.abspath('../resource/'))


if __name__ == '__main__':
    unit_test()
