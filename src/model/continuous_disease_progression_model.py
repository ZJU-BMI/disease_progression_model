# encoding=utf-8
import csv
import random
import numpy as np
import os
import datetime
from scipy.stats import norm as normal
from math import log, sqrt, exp


class ContinuousDiseaseProgressModel(object):
    def __init__(self, feature_dict, visit_time_dict, index_name_dict, parallel_sampling_time=1,
                 num_state=None, alpha=None, beta=None, gamma=None, forward_candidate=None, phi=None,
                 absorbing_state=None, backward_candidate=None, init_state_candidate=None,
                 reload_para_file_path=None):
        """
        :param feature_dict:
        :param visit_time_dict:
        :param index_name_dict:
        :param num_state:
        :param parallel_sampling_time:
        :param alpha: a tuple with two elements (shape, scale)
        :param beta: a scalar, the prior of dirichlet distribution (For the generator mat)
        :param phi: a scalar, the prior of dirichlet distribution (For the initial distribution of hidden state)
        :param gamma: dict(), the prior of observations
        :param absorbing_state: if True, the last row of generator matrix will be 0
        :param forward_candidate: the number of state that can forward jump under the current state
        :param backward_candidate: the number of state that can backward jump under the current state
        :param init_state_candidate:
        :param reload_para_file_path: None. If not none, the model will read the existing parameter directly to
        construct the object
        """
        if reload_para_file_path is not None:
            if num_state is not None and alpha is not None and beta is not None and gamma is not None \
                and forward_candidate is not None and backward_candidate is not None \
                    and init_state_candidate is not None:
                    self.__reload_path = reload_para_file_path
                    reload_para_file_path()
        else:
            if not (num_state is not None and alpha is not None and beta is not None and gamma is not None
                    and forward_candidate is not None and backward_candidate is not None
                    and init_state_candidate is not None):
                raise ValueError('Essential Parameter Lost')

            self.__num_state = num_state
            self.__init_alpha = alpha
            self.__init_beta = beta
            self.__init_phi = phi
            self.__gamma = gamma
            self.__parallel_sampling_time = parallel_sampling_time
            self.__absorbing_state = absorbing_state
            self.__forward_candidate = forward_candidate
            self.__backward_candidate = backward_candidate
            self.__init_state_candidate = init_state_candidate
            self.__reload_path = 'None'

            # used in object
            self.__observation_prob_cache = dict()

        self.__feature_dict = feature_dict
        self.__visit_time_dict = visit_time_dict
        self.__index_name_dict = index_name_dict
        self.__hidden_state_initialization()

        # init the mixture posterior set
        self.__mixture_posterior_set = dict()
        self.__prior_set = None
        for i in range(parallel_sampling_time):
            alpha_vec, beta_vec, gamma_vec, phi_vec = self.__prior_vectorization(alpha, beta, gamma, phi)
            self.__mixture_posterior_set[i] = [alpha_vec, beta_vec, gamma_vec, phi_vec]

            if i == 0:
                self.__prior_set = [alpha_vec, beta_vec, gamma_vec, phi_vec]
        print('initial accomplished')

    def __prior_vectorization(self, alpha, beta, gamma, phi):
        num_state = self.__num_state
        # vectorization the prior of Gamma distributions
        alpha_0_vec = np.full(num_state, alpha[0])
        alpha_1_vec = np.full(num_state, alpha[1])

        # vectorization the prior of dirichlet distribution (generator mat) with constraint
        beta_vec = list()
        for i in range(num_state):
            end_state = i + self.__forward_candidate
            if end_state < num_state:
                forward_state = self.__forward_candidate
            else:
                forward_state = self.__forward_candidate - (end_state - num_state + 1)

            start_state = i - self.__backward_candidate
            if start_state >= 0:
                backward_state = self.__backward_candidate
            else:
                backward_state = self.__backward_candidate + start_state
            beta_size = backward_state + forward_state
            beta_vec.append(np.full(beta_size, beta))

        # Vectorization of gamma, in fact this parameter doesn't need to vectorization. Here we just duplicate it
        gamma_vec = dict()
        for index in gamma:
            gamma_vec[index] = list()
            for item in gamma[index]:
                gamma_vec[index].append(item)

        # vectorization the prior of dirichlet distribution (init distribution)
        phi_vec = np.full(self.__init_state_candidate, phi)

        return [alpha_0_vec, alpha_1_vec], beta_vec, gamma_vec, phi_vec

    def __hidden_state_initialization(self):
        """
        we random select several time point to designate the hidden state
        :return:
        """
        parallel_sample = self.__parallel_sampling_time
        hidden_state_dict = dict()

        for index in range(parallel_sample):
            hidden_state_dict[index] = dict()
            for patient_id in self.__visit_time_dict:
                hidden_state_dict[index][patient_id] = list()

                # find the last visit
                last_visit = self.__find_last_visit_id(patient_id)
                last_visit_time = self.__visit_time_dict[patient_id][last_visit]
                slot = last_visit_time // (self.__num_state - 1)

                hidden_state_dict[index][patient_id].append([0, 0])

                for i in range(self.__num_state-1):
                    time = random.uniform(0+i*slot, i*slot+(i+1)*slot)
                    hidden_state_dict[index][patient_id].append([i + 1, time])

        self.__hidden_state = hidden_state_dict

    def optimization(self, iteration, likelihood_calculate_interval=10):

        general_start_time = start_time = datetime.datetime.now()
        for i in range(iteration):
            if i % iteration // likelihood_calculate_interval == likelihood_calculate_interval-1:
                log_likelihood = self.log_likelihood()
                end_time = datetime.datetime.now()
                time_cost = (end_time-start_time).seconds
                print('iteration {}, log likelihood: {}, time cost: {} second'
                      .format(i, log_likelihood, time_cost))
                start_time = datetime.datetime.now()

            # optimization framework
            parameter_dict = dict()
            hidden_state_dict = dict()

            # Step 1, generating new parameters and new hidden state estimation
            for j in range(self.__parallel_sampling_time):
                mixture_index = random.randint(0, self.__parallel_sampling_time-1)
                parameter_dict[j] = self.__sampling_parameters(mixture_index=mixture_index)
                hidden_state_dict[j] = self.__resampling_hidden_state(j, parameter_dict[j])

            # Step 2, updating the mixture of posterior
            self.__posterior_distribution_update(hidden_state_dict)

        general_end_time = datetime.datetime.now()
        print('Task Time Cost: {} seconds'.format((general_end_time-general_start_time).seconds))
        print('optimization accomplished')

    def __get_candidate_state_list(self, index):
        """
        get candidate jump state with preset constraint
        :param index:
        :return:
        """
        num_state = self.__num_state
        end_state = index + self.__forward_candidate
        if end_state < num_state:
            forward_state = self.__forward_candidate
        else:
            forward_state = self.__forward_candidate - (end_state - num_state + 1)

        start_state = index - self.__backward_candidate
        if start_state >= 0:
            backward_state = self.__backward_candidate
        else:
            backward_state = self.__backward_candidate + start_state
        backward_state_list = [index - j - 1 for j in range(backward_state)].__reversed__()
        forward_state_list = [index + j + 1 for j in range(forward_state)]
        state_index_list = []
        for item in backward_state_list:
            state_index_list.append(item)
        for item in forward_state_list:
            state_index_list.append(item)
        return state_index_list

    def __sampling_parameters(self, mixture_index):
        prior_set = self.__mixture_posterior_set[mixture_index]

        # init the generator matrix
        num_state = self.__num_state
        generator_mat = np.zeros([num_state, num_state])
        alpha, beta, gamma, phi = prior_set

        for i in range(len(generator_mat)):
            # sampling q_ii
            q_ii = np.random.gamma(alpha[0][i], alpha[1][i])
            generator_mat[i, i] = -q_ii

            # sampling q_ij with constraint
            jump_intensity = np.random.dirichlet(beta[i]) * - generator_mat[i, i]
            state_index_list = self.__get_candidate_state_list(i)

            for j, index in enumerate(state_index_list):
                generator_mat[i, index] = jump_intensity[j]

        if self.__absorbing_state:
            for i in range(len(generator_mat[-1])):
                generator_mat[-1, i] = 0

        # init the initial distribution vector
        init_state_prob = np.zeros([num_state])
        init_distribution = np.random.dirichlet(phi)
        for i, item in enumerate(init_distribution):
            init_state_prob[i] = item

        # init the parameter of feature distribution:
        # Note our features follows different distribution, where we need to use different method to describe the
        # prior and posterior distribution. In this study, we use three different method:
        # if a feature is discrete, we use the beta-binomial conjugate prior
        # if a feature is continuous and it follows Gaussian distribution, we use the Gaussian-Gaussian conjugate (with
        # known variance)
        # if a feature is continuous and it doesn't follow Gaussian distribution obviously, we use the Gamma conjugate
        # with known rate beta
        feature_hyperparameters = dict()
        for i in range(self.__num_state):
            single_feature_hyperparameters = dict()
            for index in gamma:
                name, distribution, prior_1, prior_2, prior_3 = gamma[index]
                if distribution == 'binomial':
                    alpha, beta = prior_1, prior_2
                    prob = np.random.beta(alpha, beta)
                    single_feature_hyperparameters[index] = {'distribution': distribution, 'prob': prob}
                elif distribution == 'gaussian':
                    mean, variance, preset_variance = prior_1, prior_2, prior_3
                    std = sqrt(variance)
                    sampled_mean = np.random.normal(mean, std)
                    single_feature_hyperparameters[index] = {'distribution': distribution,
                                                             'sampled_mean': sampled_mean,
                                                             'preset_variance': preset_variance}
                else:
                    raise ValueError('Invalid distribution name: {}'.format(distribution))
            feature_hyperparameters[i] = single_feature_hyperparameters
        return generator_mat, init_state_prob, feature_hyperparameters

    def __resampling_hidden_state(self, parallel_index, parameter_set):
        generator_mat, init_state_prob, feature_hyperparameters = parameter_set

        # generating poisson process event
        max_diagonal_ele = -1
        for i in range(len(generator_mat)):
            q_ii = - generator_mat[i, i]
            if q_ii > max_diagonal_ele:
                max_diagonal_ele = q_ii
        omega = max_diagonal_ele * 2

        previous_hidden_state = self.__hidden_state[parallel_index]
        poisson_events_dict = self.__generating_poisson_event(previous_hidden_state, generator_mat, omega)
        new_hidden_state_dict = \
            self.__blocked_gibbs_sampler(previous_hidden_state, poisson_events_dict, generator_mat, init_state_prob,
                                         feature_hyperparameters, omega)
        return new_hidden_state_dict

    def __generating_poisson_event(self, previous_hidden_state, generator_mat, omega):
        """
        :param generator_mat:
        :param omega:
        :return:
        """
        visit_time = self.__visit_time_dict

        poisson_event_dict = dict()
        for patient_id in visit_time:
            poisson_event_list = []
            
            for i in range(len(previous_hidden_state[patient_id])-1):
                markov_state_list = previous_hidden_state[patient_id]
                start_time = markov_state_list[i][1]
                end_time = markov_state_list[i+1][1]
                hidden_state = markov_state_list[i][0]
                r_t = omega + generator_mat[hidden_state, hidden_state]
                
                time = start_time
                while time < end_time:
                    new_sample_time = random.expovariate(r_t)
                    if new_sample_time + time > end_time:
                        break
                    else:
                        time = time + new_sample_time
                        poisson_event_list.append(time)

            poisson_event_dict[patient_id] = poisson_event_list
        return poisson_event_dict

    def __blocked_gibbs_sampler(self, previous_hidden_state, poisson_events_dict, generator_mat, init_state_prob,
                                feature_hyperparameters, omega):
        # combination of event
        event_sequence_dict = dict()
        for patient_id in poisson_events_dict:
            event_time_list = list()
            poisson_list = poisson_events_dict[patient_id]
            hidden_state_list = previous_hidden_state[patient_id]
            for time in poisson_list:
                event_time_list.append(time)
            for item in hidden_state_list:
                event_time_list.append(item[1])
            event_time_list = sorted(event_time_list)
            event_sequence_dict[patient_id] = event_time_list

        # forward filtering backward sampling
        transition_mat = np.identity(self.__num_state) + generator_mat/omega
        forward_procedure_dict = \
            self.__forward_procedure(event_sequence_dict, transition_mat, init_state_prob, feature_hyperparameters)
        new_hidden_state_dict \
            = self.__hidden_state_resample(forward_procedure_dict, feature_hyperparameters, transition_mat,
                                           event_sequence_dict)

        # discarding redundant event
        thined_hidden_state_dict = self.__updating_hidden_state(new_hidden_state_dict, event_sequence_dict)

        return thined_hidden_state_dict

    @ staticmethod
    def __updating_hidden_state(hidden_state_dict, event_sequence_dict):
        # combine hidden state and event sequence
        combined_sequence_dict = dict()
        for patient_id in hidden_state_dict:
            combined_sequence_dict[patient_id] = list()
            for i in range(len(hidden_state_dict[patient_id])):
                time = event_sequence_dict[patient_id][i]
                state = hidden_state_dict[patient_id][i]
                combined_sequence_dict[patient_id].append([state, time])

        # delete redundant event
        for patient_id in combined_sequence_dict:
            event_list = combined_sequence_dict[patient_id]
            index = 0
            while index < len(event_list) - 1:
                if event_list[index][0] == event_list[index+1][0]:
                    event_list.pop(index+1)
                else:
                    index += 1
            combined_sequence_dict[patient_id] = event_list
        return combined_sequence_dict

    def __log_observation_prob(self, patient_id, visit_id, hidden_state, feature_hyperparameters):
        obs_cache = self.__observation_prob_cache
        if not obs_cache.__contains__(patient_id):
            obs_cache[patient_id] = dict()
        if not obs_cache[patient_id].__contains__(visit_id):
            obs_cache[patient_id][visit_id] = dict()

        if obs_cache[patient_id][visit_id].__contains__(hidden_state):
            return obs_cache[patient_id][visit_id][hidden_state]

        feature_list = self.__feature_dict[patient_id][visit_id]
        log_observation_prob = 0
        for index, obs in enumerate(feature_list):
            if obs is None:
                continue
            obs_type = feature_hyperparameters[hidden_state][index]['distribution']
            if obs_type == 'binomial':
                prob = feature_hyperparameters[hidden_state][index]['prob']
                if obs == 1 or obs == 1.0:
                    log_observation_prob += log(prob)
                elif obs == 0 or obs == 0.0:
                    log_observation_prob += log(1 - prob)
                else:
                    raise ValueError('Data Illegal')
            elif obs_type == 'gaussian':
                mean = feature_hyperparameters[hidden_state][index]['sampled_mean']
                std = sqrt(feature_hyperparameters[hidden_state][index]['preset_variance'])
                prob_density = normal.pdf(obs, mean, std)
                if prob_density == 0:
                    log_observation_prob += -300
                else:
                    log_observation_prob += log(prob_density)
            else:
                raise ValueError('Data Illegal')

        obs_cache[patient_id][visit_id][hidden_state] = log_observation_prob
        return log_observation_prob

    def __find_visit_between_two_event(self, event_sequence_dict, patient_id, start_event, end_event):
        # find candidate visit between two visit
        visit_time_dict = self.__visit_time_dict
        current_time = event_sequence_dict[patient_id][end_event]
        last_time = event_sequence_dict[patient_id][start_event]
        candidate_visit = []
        for visit_id in visit_time_dict[patient_id]:
            visit_time = visit_time_dict[patient_id][visit_id]
            if current_time > visit_time >= last_time:
                candidate_visit.append(visit_id)
        return candidate_visit

    def __forward_procedure(self, event_sequence_dict, transition_mat, init_state_prob, feature_hyperparameters):
        forward_procedure_dict = dict()

        for patient_id in event_sequence_dict:
            forward_procedure_mat = []
            for i in range(len(event_sequence_dict[patient_id])):
                single_visit_procedure = []

                # the initial step
                if i == 0:
                    for j in range(self.__num_state):
                        if init_state_prob[j] == 0:
                            # for the case which is impossible, assign a very small value
                            single_visit_procedure.append(-1000)
                        else:
                            single_visit_procedure.append(log(init_state_prob[j]))
                    forward_procedure_mat.append(single_visit_procedure)
                    continue

                candidate_visit = self.__find_visit_between_two_event(event_sequence_dict, patient_id, i, i-1)
                for j in range(self.__num_state):
                    log_prob_j = 0

                    for k in range(self.__num_state):
                        # calculate log prob of observations
                        # find corresponding visit
                        if len(candidate_visit) == 0:
                            log_observation_prob = 0
                        else:
                            log_observation_prob = 0
                            for visit_id in candidate_visit:
                                log_observation_prob += \
                                    self.__log_observation_prob(patient_id, visit_id, k, feature_hyperparameters)

                        # calculate the log transition prob
                        transition_prob = transition_mat[k, j]
                        if transition_prob == 0 or transition_prob == 0.0:
                            log_transition_prob = -1000
                        else:
                            log_transition_prob = log(transition_prob)

                        # get the log prob of last observation
                        log_previous_prob = forward_procedure_mat[-1][k]

                        # add up
                        log_prob_j_k = log_previous_prob + log_observation_prob + log_transition_prob

                        if log_prob_j == 0:
                            log_prob_j = log_prob_j_k
                        else:
                            if log_prob_j > log_prob_j_k:
                                log_prob_j = log_prob_j + log(1+exp(log_prob_j_k-log_prob_j))
                            else:
                                log_prob_j = log_prob_j_k + log(1+exp(log_prob_j-log_prob_j_k))
                    single_visit_procedure.append(log_prob_j)
                forward_procedure_mat.append(single_visit_procedure)
            forward_procedure_dict[patient_id] = forward_procedure_mat

        return forward_procedure_dict

    def __hidden_state_resample(self, forward_procedure_dict, feature_hyperparameters, transition_mat,
                                event_sequence_dict):
        def sample_from_log_prob(log_prob_list):
            # rescaling
            candidate_state_dict = dict()
            for idx in range(self.__num_state):
                candidate_state_dict[idx] = log_prob_list[idx]

            # eliminate state which may cause numerical problem
            maximum = float('-inf')
            minimum = float('inf')
            for key in candidate_state_dict:
                log_prob = candidate_state_dict[key]
                if log_prob < minimum:
                    minimum = log_prob
                if log_prob > maximum:
                    maximum = log_prob
            while maximum - minimum > 20 and len(candidate_state_dict) > 1:
                minimum_key = None
                minimum = float('inf')
                for key in candidate_state_dict:
                    log_prob = candidate_state_dict[key]
                    if log_prob < minimum:
                        minimum_key = key
                        minimum = log_prob
                candidate_state_dict.pop(minimum_key)

                for key in candidate_state_dict:
                    log_prob = candidate_state_dict[key]
                    if log_prob < minimum:
                        minimum = log_prob

            if len(candidate_state_dict) == 1:
                for key in candidate_state_dict:
                    return key

            minimum = float('inf')
            for key in candidate_state_dict:
                if candidate_state_dict[key] < minimum:
                    minimum = candidate_state_dict[key]
            for key in candidate_state_dict:
                candidate_state_dict[key] = candidate_state_dict[key] - minimum

            # idx key map
            idx_key_dict = dict()
            idx = 0
            prob_list = []
            for key in candidate_state_dict:
                idx_key_dict[idx] = key
                prob_list.append(candidate_state_dict[key])
                idx += 1

            prob_list = np.exp(prob_list) / np.sum(np.exp(prob_list))

            cul_prob_list = []
            for idx, prob in enumerate(prob_list):
                if idx == 0:
                    cul_prob_list.append(prob_list[idx])
                else:
                    cul_prob_list.append(prob_list[idx]+cul_prob_list[idx-1])

            # sampling
            sampling_state = -1
            sampling_number = random.uniform(0, 1)
            for idx, cul_prob in enumerate(cul_prob_list):
                if cul_prob > sampling_number:
                    sampling_state = idx_key_dict[idx]
                    break
            if sampling_state == -1:
                raise ValueError('')
            return sampling_state

        hidden_state_dict = dict()
        for patient_id in forward_procedure_dict:
            hidden_state_dict[patient_id] = list()
            forward_list = forward_procedure_dict[patient_id]

            # get the last state
            last_visit = self.__find_last_visit_id(patient_id)
            log_last_prob = []
            for i in range(self.__num_state):
                log_obs_prob = self.__log_observation_prob(patient_id, last_visit, i, feature_hyperparameters)
                log_last_prob.append(log_obs_prob+forward_list[-1][i])
            state = sample_from_log_prob(log_last_prob)
            hidden_state_dict[patient_id].insert(0, state)

            # sampling state using ffbs
            i_list = [i for i in range(len(forward_list)-1)].__reversed__()
            for i in i_list:
                candidate_log_prob_list = list()
                candidate_visit = self.__find_visit_between_two_event(event_sequence_dict, patient_id, i+1, i)
                for hidden_state in range(self.__num_state):
                    next_state = hidden_state_dict[patient_id][0]
                    alpha = forward_list[i][hidden_state]
                    # calculate the log transition prob
                    transition_prob = transition_mat[hidden_state, next_state]
                    if transition_prob == 0 or transition_prob == 0.0:
                        log_transition_prob = -1000
                    else:
                        log_transition_prob = log(transition_prob)
                    log_prob_ = log_transition_prob+alpha
                    for visit in candidate_visit:
                        log_prob_ += self.__log_observation_prob(patient_id, visit, hidden_state,
                                                                 feature_hyperparameters)
                    candidate_log_prob_list.append(log_prob_)
                state = sample_from_log_prob(candidate_log_prob_list)
                hidden_state_dict[patient_id].insert(0, state)

        return hidden_state_dict

    def __posterior_distribution_update(self, hidden_state_dict):
        # evidence statistic
        sojourn_time_dict = dict()
        transition_count_dict = dict()
        observation_dict = dict()
        init_state_dict = dict()
        num_state = self.__num_state

        for parallel_index in hidden_state_dict:
            # init
            single_sojourn_time = np.zeros([num_state])
            single_init_state = np.zeros([self.__init_state_candidate])
            single_transition_count = np.zeros([num_state, num_state])
            single_observation = dict()
            for index in self.__gamma:
                single_observation[index] = [self.__gamma[index][1], []]
                for i in range(self.__num_state):
                    single_observation[index][1].append(list())

            # stat
            for patient in hidden_state_dict[parallel_index]:
                # observation stat
                observation = self.__feature_dict[patient]
                for visit_id in observation:
                    corresponding_state = self.__get_corresponding_state(parallel_index, patient, visit_id)
                    for index in range(len(observation[visit_id])):
                        if observation[visit_id][index] is not None:
                            single_observation[index][1][corresponding_state].append(observation[visit_id][index])

                # sojourn and transition stat
                hidden_state_list = self.__hidden_state[parallel_index][patient]
                for i in range(len(hidden_state_list)-1):
                    current_state = hidden_state_list[i][0]
                    next_state = hidden_state_list[i+1][0]
                    sojourn_time = hidden_state_list[i+1][1] - hidden_state_list[i][1]
                    single_transition_count[current_state, next_state] += 1
                    single_sojourn_time[current_state] += sojourn_time

                # init state stat
                single_init_state[hidden_state_list[0][0]] += 1
            sojourn_time_dict[parallel_index] = single_sojourn_time
            transition_count_dict[parallel_index] = single_transition_count
            observation_dict[parallel_index] = single_observation
            init_state_dict[parallel_index] = single_init_state

        # estimate posterior
        for parallel_index in range(self.__parallel_sampling_time):
            # update the posterior of generator mat (the diagonal elements which follows Gamma distribution)
            alpha_1_origin, alpha_2_origin = self.__prior_set[0]
            alpha_1_new_list, alpha_2_new_list = [], []
            for j in range(self.__num_state):
                # Shape Parameter
                alpha_1_new_ele = alpha_1_origin[j]
                for i in range(self.__num_state):
                    if i != j:
                        alpha_1_new_ele += transition_count_dict[parallel_index][j, i]
                # Rate Parameter
                alpha_2_new_ele = alpha_2_origin[j] + sojourn_time_dict[parallel_index][j]
                alpha_1_new_list.append(alpha_1_new_ele)
                alpha_2_new_list.append(alpha_2_new_ele)
            self.__mixture_posterior_set[parallel_index][0] = [alpha_1_new_list, alpha_2_new_list]

            # update the the posterior of generator mat (the non diagonal elements which follow Dirichlet distribution)
            beta = self.__prior_set[1]
            beta_new = list()
            for i in range(self.__num_state):
                candidate_state_list = self.__get_candidate_state_list(i)
                if len(candidate_state_list) != len(beta[i]):
                    raise ValueError('Error')
                beta_tuple = np.zeros(len(candidate_state_list))
                for j, item in enumerate(candidate_state_list):
                    beta_tuple[j] = self.__prior_set[1][i][j] + transition_count_dict[parallel_index][i, item]
                beta_new.append(beta_tuple)
            self.__mixture_posterior_set[parallel_index][1] = beta_new

            for index in observation_dict[parallel_index]:
                data_type, data = observation_dict[parallel_index][index]
                if data_type == 'gaussian':
                    # update the posterior of observations (the Gaussian-Gaussian conjugacy)
                    for hidden_state in range(num_state):
                        data_list_one_state = data[hidden_state]
                        if len(data_list_one_state) > 0:
                            obs_sum = np.sum(data_list_one_state)
                            obs_len = len(data_list_one_state)
                            prior_mean, prior_variance, preset_variance = self.__prior_set[2][index][2:]
                            posterior_mean = (prior_variance*preset_variance)/(preset_variance+obs_len*prior_variance) \
                                * (prior_mean/prior_variance + obs_sum/preset_variance)
                            posterior_variance = (preset_variance+obs_len*prior_variance) / \
                                                 (prior_variance*preset_variance)
                            self.__mixture_posterior_set[parallel_index][2][index][2] = posterior_mean
                            self.__mixture_posterior_set[parallel_index][2][index][3] = posterior_variance
                elif data_type == 'binomial':
                    # update the posterior of observations (the Bernoulli-beta conjugacy)
                    for hidden_state in range(num_state):
                        data_list_one_state = data[hidden_state]
                        if len(data_list_one_state) > 0:
                            obs_sum = np.sum(data_list_one_state)
                            obs_len = len(data_list_one_state)
                            prior_shape, prior_rate = self.__gamma[index][2:4]
                            posterior_shape = prior_shape + obs_sum
                            posterior_rate = prior_rate + obs_len - obs_sum
                            self.__mixture_posterior_set[parallel_index][2][index][2] = posterior_shape
                            self.__mixture_posterior_set[parallel_index][2][index][3] = posterior_rate

            # update the posterior of init state
            prior_phi = self.__prior_set[3]
            posterior_phi = prior_phi + init_state_dict[parallel_index]
            self.__mixture_posterior_set[parallel_index][3] = posterior_phi

    def __get_corresponding_state(self, parallel_index, patient_id, visit_id):
        visit_time = self.__visit_time_dict[patient_id][visit_id]
        hidden_state_list = self.__hidden_state[parallel_index][patient_id]
        hidden_state = -1
        for i in range(len(hidden_state_list)):
            if i == 0:
                current_time = hidden_state_list[i][1]
                if visit_time == current_time:
                    hidden_state = hidden_state_list[i][0]
            if i < len(hidden_state_list)-1:
                current_time = hidden_state_list[i][1]
                next_time = hidden_state_list[i+1][1]
                if current_time < visit_time < next_time:
                    hidden_state = hidden_state_list[i][0]
                    break
            if i == len(hidden_state_list)-1:
                current_time = hidden_state_list[i][1]
                if current_time < visit_time:
                    hidden_state = hidden_state_list[i][0]
                    break
        if hidden_state == -1:
            raise ValueError('Illegal Output')
        return hidden_state

    def log_likelihood(self):

        return 0

    def disease_state_assignment(self):
        pass

    def future_progression_estimation(self):
        pass

    def save_result(self):
        print('{}, To Be Done'.format(self.__num_state))

    def __parameter_reload(self):
        raise ValueError('To Be Done')

    def __find_last_visit_id(self, patient_id):
        # find last visit
        last_visit = -1
        last_time = -1
        for visit_id in self.__visit_time_dict[patient_id]:
            visit_time = self.__visit_time_dict[patient_id][visit_id]
            if visit_time > last_time:
                last_time = visit_time
                last_visit = visit_id
        return last_visit


def read_prior(file_name):
    gamma_dict = dict()
    with open(file_name, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for i, line in enumerate(csv_reader):
            if i == 0:
                continue
            index, name, distribution, prior_1, prior_2, prior_3 = line
            if prior_3 == 'None':
                gamma_dict[int(index)] = name, distribution, float(prior_1), float(prior_2), None
            else:
                gamma_dict[int(index)] = name, distribution, float(prior_1), float(prior_2), float(prior_3)
    return gamma_dict


def read_data(file_name, limitation=None):
    feature_dict = dict()
    visit_time_dict = dict()
    index_name_dict = dict()
    with open(file_name, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for i, line in enumerate(csv_reader):
            if i == 0:
                for j, item in enumerate(line[3:]):
                    index_name_dict[j] = item
                continue

            patient_id, visit_id, visit_time = line[0], int(line[1]), int(line[2])
            if not feature_dict.__contains__(patient_id):
                if limitation is not None:
                    if len(feature_dict) > limitation:
                        break
                feature_dict[patient_id] = dict()
            feature_dict[patient_id][visit_id] = list()
            for item in line[3:]:
                if item is None or len(item) == 0:
                    feature_dict[patient_id][visit_id].append(None)
                else:
                    feature_dict[patient_id][visit_id].append(float(item))
            if not visit_time_dict.__contains__(patient_id):
                visit_time_dict[patient_id] = dict()
            visit_time_dict[patient_id][visit_id] = visit_time
    return feature_dict, visit_time_dict, index_name_dict


def unit_test():
    limitation = None
    num_state = 5
    alpha = [1, 0.001]
    beta = 0.1
    gamma_file_name = os.path.abspath('../../resource/特征先验.csv')
    gamma_dict = read_prior(gamma_file_name)
    phi = 0.1

    parallel_sampling_time = 3
    forward_candidate = 2
    backward_candidate = 2
    absorbing_state = False
    init_state_candidate = 3

    data_file_name = \
        os.path.abspath('../../resource/未预处理长期纵向数据_离散化_False_特别筛选_True_截断年份_2009.csv')
    feature_dict, visit_time_dict, index_name_dict = read_data(data_file_name, limitation=limitation)

    cdpm = ContinuousDiseaseProgressModel(feature_dict, visit_time_dict, index_name_dict,
                                          parallel_sampling_time=parallel_sampling_time, num_state=num_state,
                                          alpha=alpha, beta=beta, gamma=gamma_dict, phi=phi,
                                          forward_candidate=forward_candidate, backward_candidate=backward_candidate,
                                          init_state_candidate=init_state_candidate, absorbing_state=absorbing_state)
    cdpm.optimization(1)
    cdpm.save_result()


if __name__ == '__main__':
    unit_test()
