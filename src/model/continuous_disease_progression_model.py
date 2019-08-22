# encoding=utf-8
import csv
import random
import numpy as np
import os
import datetime
from scipy.stats import norm as normal
from scipy.linalg import expm
from math import log, sqrt, exp


class ContinuousDiseaseProgressModel(object):
    def __init__(self, obs_dict, visit_time_dict, index_name_dict, parallel_sampling_time=1,
                 num_state=None, alpha=None, beta=None, gamma=None, forward_candidate=1, phi=None,
                 backward_candidate=0, init_state_candidate=None, reload_para_file_path=None):
        """
        :param obs_dict:
        :param visit_time_dict: {patient: {visit_id: time}}
        :param index_name_dict: to recover the name of observation given index
        :param num_state:number of hidden state
        :param parallel_sampling_time: scalar, used in generating the new hidden state
        :param alpha: a tuple with two elements (shape, scale)
        :param beta: a scalar, the prior of dirichlet distribution (For the generator mat)
        :param phi: a scalar, the prior of dirichlet distribution (For the initial distribution of hidden state)
        :param gamma: dict(), the prior of observations
        :param forward_candidate: the number of state that can forward jump under the current state
        :param backward_candidate: the number of state that can backward jump under the current state
        :param init_state_candidate: the candidate state of the first observation
        :param reload_para_file_path: None. If not none, the model will read the existing parameter directly to
        construct the object
        """
        if reload_para_file_path is not None:
            if num_state is not None and alpha is not None and beta is not None and gamma is not None \
                and forward_candidate is not None and backward_candidate is not None \
                    and init_state_candidate is not None:
                    self.__reload_path = reload_para_file_path
                    self.__parameter_reload()
        else:
            if not (num_state is not None and alpha is not None and beta is not None and gamma is not None
                    and forward_candidate is not None and backward_candidate is not None
                    and init_state_candidate is not None):
                raise ValueError('Essential Parameter Lost')

            self.__num_state = num_state
            self.__init_alpha = alpha
            self.__init_beta = beta
            self.__init_phi = phi
            self.__init_gamma = gamma
            self.__parallel_sampling_time = parallel_sampling_time
            self.__forward_candidate = forward_candidate
            self.__backward_candidate = backward_candidate
            self.__init_state_candidate = init_state_candidate
            self.__reload_path = 'None'
            self.__iteration = None
            self.__log_likelihood_trend = list()

        # mediate variable
        self.__sojourn_time_dict = None
        self.__transition_count_dict = None
        self.__observation_count_dict = None
        self.__init_state_dict = None

        # feed data
        self.__obs_dict = obs_dict
        self.__visit_time_dict = visit_time_dict
        self.__index_name_dict = index_name_dict

        # initialization
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
            state_list = self.__get_candidate_state_list(i)
            beta_vec.append(np.full(len(state_list), beta))

        # Vectorization of gamma, in fact this parameter doesn't need to vectorization. Here we just duplicate it
        gamma_vec = dict()
        for index in gamma:
            gamma_vec[index] = dict()
            for state in range(self.__num_state):
                gamma_vec[index][state] = list()
                for item in gamma[index]:
                    gamma_vec[index][state].append(item)

        # vectorization the prior of dirichlet distribution (init distribution)
        phi_vec = np.full(self.__init_state_candidate, phi)

        return [alpha_0_vec, alpha_1_vec], beta_vec, gamma_vec, phi_vec

    def __hidden_state_initialization(self):
        """
        we randomly select several time point to designate the hidden state
        :return:
        """
        parallel_sample = self.__parallel_sampling_time
        hidden_state_dict = dict()

        for parallel_index in range(parallel_sample):
            hidden_state_dict[parallel_index] = dict()
            for patient_id in self.__visit_time_dict:
                hidden_state_dict[parallel_index][patient_id] = list()

                # find the last visit
                last_visit = self.__find_last_visit_id(patient_id)
                last_visit_time = self.__visit_time_dict[patient_id][last_visit]
                slot = last_visit_time // (self.__num_state - 1)

                # we will assign the first visit as state 0
                hidden_state_dict[parallel_index][patient_id].append([0, 0])

                for i in range(self.__num_state-1):
                    time = random.uniform(i*slot, (i+1)*slot)
                    hidden_state_dict[parallel_index][patient_id].append([i + 1, time])

        self.__hidden_state = hidden_state_dict

    def optimization(self, iteration, likelihood_calculate_interval=10):
        self.__iteration = iteration
        general_start_time = start_time = datetime.datetime.now()

        # The Convergence of this optimization is proofed in
        # The Calculation of Posterior Distributions by Data Augmentation, Tanner, 1987
        for i in range(0, iteration):
            if i == iteration or i % likelihood_calculate_interval == 0:
                visit_time_dict = self.__visit_time_dict
                log_likelihood = self.log_likelihood(visit_time_dict)
                end_time = datetime.datetime.now()
                time_cost = (end_time-start_time).seconds
                print('iteration {}, log likelihood: {}, time cost: {} second'
                      .format(i, log_likelihood, time_cost))
                start_time = datetime.datetime.now()

                self.__log_likelihood_trend.append([i, log_likelihood])

            # optimization framework
            # Step 1, generating new parameters and new hidden state estimation
            for j in range(self.__parallel_sampling_time):
                mixture_index = random.randint(0, self.__parallel_sampling_time-1)
                parameters = self.__sampling_parameters(mixture_index=mixture_index)
                self.__hidden_state[j] = self.__blocked_gibbs_sampler(j, parameters)

            # Step 2, updating the mixture of posterior
            self.__posterior_distribution_update()

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
            backward_state = index

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

            state_index_list = self.__get_candidate_state_list(i)
            if len(state_index_list) > 0:
                # sampling q_ij with constraint
                jump_intensity = np.random.dirichlet(beta[i]) * - generator_mat[i, i]

                for j, index in enumerate(state_index_list):
                    generator_mat[i, index] = jump_intensity[j]
            elif len(state_index_list) == 0:
                # means the index is an absorbing state
                generator_mat[i, i] = 0

        # init the initial distribution vector
        init_state_prob = np.zeros([num_state])
        init_distribution = np.random.dirichlet(phi)
        for i, item in enumerate(init_distribution):
            init_state_prob[i] = item

        # init the parameter of obs distribution:
        # Note our obs follows different distribution, where we need to use different method to describe the
        # prior and posterior distribution. In this study, we use three different method:
        # if a obs is discrete, we use the beta-binomial conjugate prior
        # if a obs is continuous and it follows Gaussian distribution, we use the Gaussian-Gaussian conjugate (with
        # known variance)
        # if a obs is continuous and it doesn't follow Gaussian distribution obviously, we use the Gamma conjugate
        # with known rate beta
        obs_hyperparameters = dict()
        for index in gamma:
            obs_hyperparameters[index] = dict()
            for state in gamma[index]:
                name, distribution, prior_1, prior_2, prior_3 = gamma[index][state]
                if distribution == 'binomial':
                    alpha, beta = prior_1, prior_2
                    prob = np.random.beta(alpha, beta)
                    obs_hyperparameters[index][state] = {'distribution': distribution, 'prob': prob}
                elif distribution == 'gaussian':
                    mean, variance, preset_variance = prior_1, prior_2, prior_3
                    std = sqrt(variance)
                    sampled_mean = np.random.normal(mean, std)
                    obs_hyperparameters[index][state] = {'distribution': distribution,
                                                         'sampled_mean': sampled_mean,
                                                         'preset_variance': preset_variance}
                else:
                    raise ValueError('Invalid distribution name: {}'.format(distribution))
        return generator_mat, init_state_prob, obs_hyperparameters

    def __blocked_gibbs_sampler(self, parallel_index, parameter_set):
        generator_mat, init_state_prob, obs_hyperparameters = parameter_set

        # Step 1: generating poisson process event
        max_diagonal_ele = -1
        for i in range(len(generator_mat)):
            q_ii = - generator_mat[i, i]
            if q_ii > max_diagonal_ele:
                max_diagonal_ele = q_ii
        # omega = max_diagonal_ele * 2 follows the setting in Fast MCMC Sampling for Markov Jump Processes and
        # Extensions (Figure 3 left of that paper)
        omega = max_diagonal_ele * 2
        previous_hidden_state = self.__hidden_state[parallel_index]
        poisson_events_dict = self.__generating_poisson_event(previous_hidden_state, generator_mat, omega)

        # Step 2: Sample the hidden state using FFBS method
        new_hidden_state_dict = \
            self.__forward_filtering_backward_sampling(previous_hidden_state, poisson_events_dict, generator_mat,
                                                       init_state_prob, obs_hyperparameters, omega)
        # for test usage
        # hidden_state_count = 0
        # for patient_id in new_hidden_state_dict:
        #     hidden_state_count += len(new_hidden_state_dict[patient_id])
        # print('average hidden states in a trajectory: {}'.format(hidden_state_count/len(new_hidden_state_dict)))
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
            
            for i in range(len(previous_hidden_state[patient_id])):
                markov_state_list = previous_hidden_state[patient_id]
                start_time = markov_state_list[i][1]
                if i == len(previous_hidden_state[patient_id]) - 1:
                    last_visit = self.__find_last_visit_id(patient_id)
                    end_time = visit_time[patient_id][last_visit]
                else:
                    end_time = markov_state_list[i+1][1]
                hidden_state = markov_state_list[i][0]
                r_t = omega + generator_mat[hidden_state, hidden_state]
                
                time = start_time
                while time < end_time:
                    # expovariate generate the event time which follows Poisson process
                    new_sample_time = random.expovariate(r_t)
                    if new_sample_time + time > end_time:
                        break
                    else:
                        time = time + new_sample_time
                        poisson_event_list.append(time)

            poisson_event_dict[patient_id] = poisson_event_list
        return poisson_event_dict

    def __forward_filtering_backward_sampling(self, previous_hidden_state, poisson_events_dict, generator_mat,
                                              init_state_prob, obs_hyperparameters, omega):
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
            self.__homogeneous_forward_procedure(event_sequence_dict, transition_mat, init_state_prob,
                                                 obs_hyperparameters)
        new_hidden_state_dict \
            = self.__hidden_state_resample(forward_procedure_dict, obs_hyperparameters, transition_mat,
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

    @staticmethod
    # checked
    def __log_observation_prob(obs_list=None, hidden_state=None, obs_hyperparameters=None):
        """
        calculate the observation of given condition.
        :param hidden_state:
        :return:
        """

        log_observation_prob = 0
        for index, obs in enumerate(obs_list):
            if obs is None:
                continue

            obs_type = obs_hyperparameters[index][hidden_state]['distribution']
            if obs_type == 'binomial':
                prob = obs_hyperparameters[index][hidden_state]['prob']
                if obs == 1 or obs == 1.0:
                    log_observation_prob += log(prob)
                elif obs == 0 or obs == 0.0:
                    log_observation_prob += log(1 - prob)
                else:
                    raise ValueError('Data Illegal')
            elif obs_type == 'gaussian':
                mean = obs_hyperparameters[index][hidden_state]['sampled_mean']
                std = sqrt(obs_hyperparameters[index][hidden_state]['preset_variance'])
                prob_density = normal.pdf(obs, mean, std)
                if prob_density == 0:
                    log_observation_prob += -400
                else:
                    log_observation_prob += log(prob_density)
            else:
                raise ValueError('Data Illegal')

        return log_observation_prob

    def __find_visit_between_two_event(self, patient_id, current_time, last_time):
        # find candidate visit between two visit
        # if happen time of a visit equals to the happen time of end event, we think the event is not valid
        visit_time_dict = self.__visit_time_dict
        candidate_visit = []
        if current_time is None:
            current_time = 10000000
        for visit_id in visit_time_dict[patient_id]:
            visit_time = visit_time_dict[patient_id][visit_id]
            if current_time > visit_time >= last_time:
                candidate_visit.append(visit_id)
        return candidate_visit

    def __homogeneous_forward_procedure(self, event_sequence_dict, transition_mat, init_state_prob,
                                        obs_hyperparameters):
        """
        Note, in this function alpha_j(k) doesn't conclude the observation at k
        """
        forward_procedure_dict = dict()

        for patient_id in event_sequence_dict:
            forward_procedure_mat = []
            for i in range(len(event_sequence_dict[patient_id])):
                single_visit_procedure = []

                # the initial step
                # we assume first visit is with the first
                if i == 0:
                    for j in range(self.__num_state):
                        if init_state_prob[j] == 0:
                            # for the case which is impossible, assign a very small value
                            single_visit_procedure.append(-10000000)
                        else:
                            single_visit_procedure.append(log(init_state_prob[j]))
                    forward_procedure_mat.append(single_visit_procedure)
                    continue

                # find corresponding visit
                current_time = event_sequence_dict[patient_id][i]
                previous_time = event_sequence_dict[patient_id][i-1]
                candidate_visit = self.__find_visit_between_two_event(patient_id, current_time, previous_time)
                log_observation_prob_list = list()
                for j in range(self.__num_state):
                    # calculate log prob of observations
                    if len(candidate_visit) == 0:
                        log_observation_prob = 0
                    else:
                        log_observation_prob = 0
                        for visit_id in candidate_visit:
                            obs_list = self.__obs_dict[patient_id][visit_id]
                            log_observation_prob += self.__log_observation_prob(obs_list=obs_list, hidden_state=j,
                                                                                obs_hyperparameters=obs_hyperparameters)
                    log_observation_prob_list.append(log_observation_prob)

                for j in range(self.__num_state):
                    # j is the current state
                    log_prob_j = 0
                    for k in range(self.__num_state):
                        # k is the previous state
                        # calculate the log transition prob
                        transition_prob = transition_mat[k, j]
                        if transition_prob == 0 or transition_prob == 0.0:
                            log_transition_prob = -10000000
                        else:
                            log_transition_prob = log(transition_prob)

                        # get the log prob of last observation
                        log_previous_prob = forward_procedure_mat[-1][k]

                        # add up
                        log_prob_j_k = log_previous_prob + log_observation_prob_list[k] + log_transition_prob

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

    def __hidden_state_resample(self, forward_procedure_dict, obs_hyperparameters, transition_mat,
                                event_sequence_dict):
        """
        Note, according to our settings. The forward procedure doesn't contain the current observation
        and the last observation (may be more event) is at the same time as the last event
        :param forward_procedure_dict:
        :param obs_hyperparameters:
        :param transition_mat:
        :param event_sequence_dict:
        :return:
        """
        hidden_state_dict = dict()
        for patient_id in forward_procedure_dict:
            hidden_state_dict[patient_id] = list()
            forward_list = forward_procedure_dict[patient_id]

            # get the last state
            previous_time = event_sequence_dict[patient_id][-1]
            visit_list = self.__find_visit_between_two_event(patient_id, None, previous_time)
            log_last_prob = []
            if len(visit_list) == 0:
                raise ValueError('')
            for i in range(self.__num_state):
                log_obs_prob = 0
                for visit_id in visit_list:
                    obs_list = self.__obs_dict[patient_id][visit_id]

                    log_obs_prob += self.__log_observation_prob(obs_list=obs_list, hidden_state=i,
                                                                obs_hyperparameters=obs_hyperparameters)
                log_last_prob.append(log_obs_prob+forward_list[-1][i])
            state = self.__sample_from_log_prob(log_last_prob)
            hidden_state_dict[patient_id].insert(0, state)

            # sampling state using ffbs
            i_list = [i for i in range(len(forward_list)-1)].__reversed__()
            for i in i_list:
                candidate_log_prob_list = list()
                current_time = event_sequence_dict[patient_id][i]
                previous_time = event_sequence_dict[patient_id][i-1]
                candidate_visit = self.__find_visit_between_two_event(patient_id, current_time, previous_time)
                for hidden_state in range(self.__num_state):
                    next_state = hidden_state_dict[patient_id][0]
                    alpha = forward_list[i][hidden_state]
                    # calculate the log transition prob
                    transition_prob = transition_mat[hidden_state, next_state]
                    if transition_prob == 0 or transition_prob == 0.0:
                        log_transition_prob = -10000000
                    else:
                        log_transition_prob = log(transition_prob)
                    log_prob_ = log_transition_prob+alpha

                    # calculate the log prob of observation
                    log_obs_prob = 0
                    for visit in candidate_visit:
                        obs_list = self.__obs_dict[patient_id][visit]
                        log_obs_prob += self.__log_observation_prob(hidden_state=hidden_state, obs_list=obs_list,
                                                                    obs_hyperparameters=obs_hyperparameters)
                    log_prob_ += log_obs_prob
                    candidate_log_prob_list.append(log_prob_)
                state = self.__sample_from_log_prob(candidate_log_prob_list)
                hidden_state_dict[patient_id].insert(0, state)
        return hidden_state_dict

    def __sample_from_log_prob(self, log_prob_list):
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
        while maximum - minimum > 40 and len(candidate_state_dict) > 1:
            minimum_key = None
            minimum = float('inf')
            for key in candidate_state_dict:
                log_prob = candidate_state_dict[key]
                if log_prob < minimum:
                    minimum_key = key
                    minimum = log_prob
            candidate_state_dict.pop(minimum_key)

            minimum = float('inf')
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
                cul_prob_list.append(prob_list[idx] + cul_prob_list[idx - 1])

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

    def __posterior_distribution_update(self):
        # evidence statistic
        sojourn_time_dict = dict()
        transition_count_dict = dict()
        observation_dict = dict()
        init_state_dict = dict()
        num_state = self.__num_state
        hidden_state_dict = self.__hidden_state

        for parallel_index in hidden_state_dict:
            # init
            single_sojourn_time = np.zeros([num_state])
            single_init_state = np.zeros([self.__init_state_candidate])
            single_transition_count = np.zeros([num_state, num_state])
            single_observation = dict()
            for index in self.__init_gamma:
                single_observation[index] = [self.__init_gamma[index][1], []]
                for i in range(self.__num_state):
                    single_observation[index][1].append(list())

            # stat
            for patient in hidden_state_dict[parallel_index]:
                # observation stat
                observation = self.__obs_dict[patient]
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

        self.__sojourn_time_dict = sojourn_time_dict
        self.__transition_count_dict = transition_count_dict
        self.__observation_count_dict = observation_dict
        self.__init_state_dict = init_state_dict

        # estimate posterior
        for parallel_index in range(self.__parallel_sampling_time):
            # update the posterior of generator mat (the diagonal elements which follows Gamma distribution)
            # we use the inverse rate because np.random.gamma use the inverse rate as parameter rather than rate
            alpha_1_origin, alpha_2_origin = self.__prior_set[0][0], 1 / self.__prior_set[0][1]
            alpha_1_new_list, alpha_2_new_list = [], []
            for j in range(self.__num_state):
                # Shape Parameter
                alpha_1_new_ele = alpha_1_origin[j]
                for i in range(self.__num_state):
                    if i != j:
                        alpha_1_new_ele += transition_count_dict[parallel_index][j, i]
                # Scale Parameter
                alpha_2_new_ele = alpha_2_origin[j] + sojourn_time_dict[parallel_index][j]
                alpha_2_new_ele = 1 / alpha_2_new_ele
                alpha_1_new_list.append(alpha_1_new_ele)
                alpha_2_new_list.append(alpha_2_new_ele)
            alpha_1_new_list = np.array(alpha_1_new_list)
            alpha_2_new_list = np.array(alpha_2_new_list)
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

            # update the posterior of observation
            for index in observation_dict[parallel_index]:
                data_type, data = observation_dict[parallel_index][index]
                if data_type == 'gaussian':
                    # update the posterior of observations (the Gaussian-Gaussian conjugacy)
                    for hidden_state in range(num_state):
                        data_list_one_state = data[hidden_state]
                        if len(data_list_one_state) > 0:
                            obs_sum = np.sum(data_list_one_state)
                            obs_len = len(data_list_one_state)
                            prior_mean, prior_variance, preset_variance = self.__prior_set[2][index][hidden_state][2:]
                            posterior_variance = preset_variance
                            variance_ = (prior_variance*preset_variance)/(preset_variance+obs_len*prior_variance)
                            posterior_mean = variance_*(prior_mean/prior_variance + obs_sum/preset_variance)
                            self.__mixture_posterior_set[parallel_index][2][index][hidden_state][2] = posterior_mean
                            self.__mixture_posterior_set[parallel_index][2][index][hidden_state][3] = posterior_variance
                elif data_type == 'binomial':
                    # update the posterior of observations (the Bernoulli-beta conjugacy)
                    for hidden_state in range(num_state):
                        data_list_one_state = data[hidden_state]
                        if len(data_list_one_state) > 0:
                            obs_sum = np.sum(data_list_one_state)
                            obs_len = len(data_list_one_state)
                            prior_alpha, prior_beta = self.__prior_set[2][index][hidden_state][2:4]
                            posterior_alpha = prior_alpha + obs_sum
                            posterior_beta = prior_beta + obs_len - obs_sum
                            self.__mixture_posterior_set[parallel_index][2][index][hidden_state][2] = posterior_alpha
                            self.__mixture_posterior_set[parallel_index][2][index][hidden_state][3] = posterior_beta

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
                    break
            if i < len(hidden_state_list)-1:
                current_time = hidden_state_list[i][1]
                next_time = hidden_state_list[i+1][1]
                if current_time <= visit_time < next_time:
                    hidden_state = hidden_state_list[i][0]
                    break
            if i == len(hidden_state_list)-1:
                current_time = hidden_state_list[i][1]
                if current_time < visit_time:
                    hidden_state = hidden_state_list[i][0]
                    break
        if hidden_state == -1:
            raise ValueError('')
        return hidden_state

    # checked
    def log_likelihood(self, visit_time_dict):
        # Estimate the parameter. We use the mean of mixture distribution as the parameter
        generate_mat, init_vec, obs_para_dict = self.__estimate_parameter()
        forward_procedure_dict = self.__inhomogeneous_forward_procedure(visit_time_dict, generate_mat, init_vec,
                                                                        obs_para_dict)

        general_log_likelihood = 0
        for patient_id in forward_procedure_dict:
            patient_likelihood_list = forward_procedure_dict[patient_id][-1]
            log_prob = 0
            for item in patient_likelihood_list:
                if log_prob == 0:
                    log_prob = item
                else:
                    if log_prob > item:
                        log_prob = log_prob + log(1 + exp(item - log_prob))
                    else:
                        log_prob = item + log(1 + exp(log_prob - item))
            general_log_likelihood += log_prob
        return general_log_likelihood

    # checked
    def __inhomogeneous_forward_procedure(self, visit_time_dict, generate_mat, init_vec, obs_para_dict):
        """
        in this function, we follow the settings we used in __homogeneous_forward_procedure
        That is, the alpha_j(t) doesn't contain the current output
        """

        # rearrange the visit time dict because the visit time dict is not ordered, which may cause problem
        ordered_visit_time_dict = dict()
        for patient_id in visit_time_dict:
            ordered_visit_list = list()
            for visit_id in visit_time_dict[patient_id]:
                ordered_visit_list.append([visit_id, visit_time_dict[patient_id][visit_id]])
            ordered_visit_list.sort(key=lambda x: x[0])
            ordered_visit_time_dict[patient_id] = ordered_visit_list

        # inhomogeneous forward procedure
        forward_procedure_dict = dict()
        for patient_id in visit_time_dict:
            forward_procedure_mat = []
            for i in range(len(visit_time_dict[patient_id])):
                single_visit_procedure = []
                visit_id = ordered_visit_time_dict[patient_id][i][0]

                # the initial step
                if i == 0:
                    for j in range(self.__num_state):
                        if init_vec[j] == 0:
                            # for the case which is impossible, assign a very small value
                            single_visit_procedure.append(-10000000)
                        else:
                            single_visit_procedure.append(log(init_vec[j]))
                    forward_procedure_mat.append(single_visit_procedure)
                    continue

                # calculate the log likelihood of observation
                single_obs_cache = list()
                previous_visit_id = ordered_visit_time_dict[patient_id][i-1][0]
                for k in range(self.__num_state):
                    # calculate log prob of observations
                    # find corresponding visit
                    obs_list = self.__obs_dict[patient_id][previous_visit_id]
                    log_obs_prob = self.__log_observation_prob(hidden_state=k, obs_hyperparameters=obs_para_dict,
                                                               obs_list=obs_list)
                    single_obs_cache.append(log_obs_prob)

                # calculate the log transition prob
                time_interval = visit_time_dict[patient_id][visit_id] - visit_time_dict[patient_id][previous_visit_id]
                transition_mat = expm(generate_mat * time_interval)

                for j in range(self.__num_state):
                    # j is the current state
                    log_prob_j = 0

                    for k in range(self.__num_state):
                        # j is the previous state
                        transition_prob = transition_mat[k, j]
                        if transition_prob == 0 or transition_prob == 0.0:
                            log_transition_prob = -10000000
                        else:
                            log_transition_prob = log(transition_prob)

                        # get the log prob of last observation
                        log_previous_prob = forward_procedure_mat[-1][k]

                        # add up
                        log_prob_j_k = log_previous_prob + single_obs_cache[k] + log_transition_prob

                        if log_prob_j == 0:
                            log_prob_j = log_prob_j_k
                        else:
                            if log_prob_j > log_prob_j_k:
                                log_prob_j = log_prob_j + log(1 + exp(log_prob_j_k - log_prob_j))
                            else:
                                log_prob_j = log_prob_j_k + log(1 + exp(log_prob_j - log_prob_j_k))
                    single_visit_procedure.append(log_prob_j)
                forward_procedure_mat.append(single_visit_procedure)

            # calculate the last prob
            single_visit_procedure = []
            single_obs_cache = list()
            last_visit_id = ordered_visit_time_dict[patient_id][-1][0]
            for k in range(self.__num_state):
                # calculate log prob of observations
                # find corresponding visit
                obs_list = self.__obs_dict[patient_id][last_visit_id]
                log_obs_prob = self.__log_observation_prob(hidden_state=k, obs_hyperparameters=obs_para_dict,
                                                           obs_list=obs_list)
                single_obs_cache.append(log_obs_prob)

            for k in range(self.__num_state):
                # j is the previous state
                # get the log prob of last observation
                log_previous_prob = forward_procedure_mat[-1][k]
                # add up
                log_prob_j = log_previous_prob + single_obs_cache[k]
                single_visit_procedure.append(log_prob_j)
            forward_procedure_mat.append(single_visit_procedure)
            forward_procedure_dict[patient_id] = forward_procedure_mat
        return forward_procedure_dict

    def __estimate_parameter(self):
        num_state = self.__num_state
        mixture_posterior_set = self.__mixture_posterior_set
        num_component = self.__parallel_sampling_time

        alpha_shape_list, alpha_scale_list, beta_dirichlet_list, gamma_set_list, phi_dirichlet_list = [], [], [], [], []
        for parallel_index in mixture_posterior_set:
            alpha_shape, alpha_scale = mixture_posterior_set[parallel_index][0]
            beta_dirichlet = mixture_posterior_set[parallel_index][1]
            gamma_set = mixture_posterior_set[parallel_index][2]
            phi_dirichlet = mixture_posterior_set[parallel_index][3]

            alpha_shape_list.append(alpha_shape)
            alpha_scale_list.append(alpha_scale)
            beta_dirichlet_list.append(beta_dirichlet)
            gamma_set_list.append(gamma_set)
            phi_dirichlet_list.append(phi_dirichlet)

        # estimate alpha
        alpha_mean_list = list()
        for i in range(num_state):
            alpha_mean = 0
            for j in range(num_component):
                shape = alpha_shape_list[j][i]
                scale = alpha_scale_list[j][i]
                alpha_mean += 1 / num_component * shape * scale
            alpha_mean_list.append(alpha_mean)

        # estimate beta
        beta_mean_list = list()
        for i in range(num_state):
            beta_mean = np.zeros(len(beta_dirichlet_list[0][i]))
            for j in range(num_component):
                beta_mean += 1 / num_component * \
                             np.array(beta_dirichlet_list[j][i]) / np.sum(beta_dirichlet_list[j][i])
            beta_mean_list.append(beta_mean)

        # estimate gamma
        gamma_mean_dict = dict()
        for feature_idx in gamma_set_list[0]:
            gamma_mean_dict[feature_idx] = dict()
            for hidden_state in gamma_set_list[0][feature_idx]:
                if gamma_set_list[0][feature_idx][hidden_state][1] == 'gaussian':
                    # the Variance of mixture of Gaussian is difficult to calculate.
                    # Therefore, we directly use the mean and variance of the first component
                    gamma_mean = {'distribution': 'gaussian',
                                  'sampled_mean': gamma_set_list[0][feature_idx][hidden_state][2],
                                  'preset_variance': gamma_set_list[0][feature_idx][hidden_state][3]}
                elif gamma_set_list[0][feature_idx][hidden_state][1] == 'binomial':
                    gamma_mean_prob = 0
                    for i in range(num_component):
                        a = gamma_set_list[i][feature_idx][hidden_state][2]
                        b = gamma_set_list[i][feature_idx][hidden_state][3]
                        gamma_mean_prob += 1 / num_component * a / (a + b)
                    gamma_mean = {'distribution': 'binomial', 'prob': gamma_mean_prob}
                else:
                    raise ValueError('')
                gamma_mean_dict[feature_idx][hidden_state] = gamma_mean

        # estimate phi
        phi_mean = np.zeros([self.__init_state_candidate])
        for i in range(num_component):
            phi_mean += 1 / num_component * np.array(phi_dirichlet_list[i]) / np.sum(phi_dirichlet_list[i])

        # construct parameter
        generate_mat = np.zeros([num_state, num_state])
        for i in range(num_state):
            generate_mat[i, i] = -1 * alpha_mean_list[i]
            candidate_state_list = self.__get_candidate_state_list(i)
            if len(candidate_state_list) != len(beta_mean_list[i]):
                raise ValueError('')
            if len(candidate_state_list) != 0:
                for j, item in enumerate(candidate_state_list):
                    generate_mat[i, item] = beta_mean_list[i][j] * -1 * generate_mat[i, i]
            else:
                generate_mat[i, i] = 0

        init_vec = np.zeros([num_state])
        for i, item in enumerate(phi_mean):
            init_vec[i] = item
        obs_dict = gamma_mean_dict
        return generate_mat, init_vec, obs_dict

    def disease_state_assignment(self, obs_dict, time_dict):
        """
        Inference the hidden states of a given trajectory list set using Viterbi Algorithm,
        which was introduced in the section 3.B of 'A tutorial on hidden markov models and selected
        applications in speech recognition', and we follow the notations used in this article.
        :param obs_dict: The data structure of 'new_trajectory_list'
        is same as the corresponding parameter of constructor:
        :param time_dict
        :return: hidden state estimation:
            level 1: dict of patient trajectories [id_1:trajectory_1, id_2: trajectory_2, ..., id_M: trajectory_M]
            level 2: For each trajectory of patient i, the structure is : [state_1, state_2, ..., state_T_i]
        """

        def viterbi_algorithm(trajectory, generate_matrix, init_vector, obs_parameter_dict):
            # delta indicates the maximum probability
            num_state = self.__num_state
            delta = np.zeros([len(trajectory), num_state])
            psi = np.zeros([len(trajectory), num_state])

            for j, single_visit in enumerate(trajectory):
                # to avoid the underflow problem, we use the log likelihood

                # calculate the observation probability
                observation_log_prob_mat = np.zeros([num_state])
                for hidden_state in range(self.__num_state):
                    # calculate log prob of observations
                    # find corresponding visit
                    observation_log_prob_mat[hidden_state] = \
                        self.__log_observation_prob(obs_list=single_visit[2], hidden_state=hidden_state,
                                                    obs_hyperparameters=obs_parameter_dict)

                # init step
                if j == 0:
                    for k in range(num_state):
                        log_prob = observation_log_prob_mat[k]

                        if init_vector[k] == 0:
                            log_prob += -10000000
                        else:
                            log_prob += log(init_vector[k])

                        delta[j, k] = log_prob
                        psi[j, k] = -10000
                    continue

                # recursion step
                time_interval = trajectory[j][1] - trajectory[j-1][1]
                transition_mat = expm(generate_matrix*time_interval)
                for k in range(num_state):
                    max_prob = -float('inf')
                    for prev_state in range(num_state):
                        # To avoid the numerical unstable problem when we constraint the state transition direction
                        if transition_mat[prev_state, k] == 0:
                            log_prob = -10000000
                        else:
                            log_prob = delta[j - 1, prev_state] + log(transition_mat[prev_state, k]) + \
                                       observation_log_prob_mat[k]

                        if log_prob > max_prob:
                            max_prob = log_prob
                            delta[j, k] = log_prob
                            psi[j, k] = prev_state

            # terminate step
            estimated_state_list = list()
            estimated_last_state = np.argmax(delta[-1])
            estimated_state_list.insert(0, estimated_last_state)
            for i in range(len(trajectory) - 1):
                estimated_last_state = int(psi[len(trajectory) - i - 1, estimated_last_state])
                estimated_state_list.insert(0, estimated_last_state)
            return estimated_state_list

        generate_mat, init_vec, obs_para_dict = self.__estimate_parameter()
        hidden_state_dict = dict()

        # obs time fusion
        trajectory_dict = dict()
        for patient_id in obs_dict:
            trajectory_list = list()
            for visit_id in obs_dict[patient_id]:
                obs = obs_dict[patient_id][visit_id]
                visit_time = time_dict[patient_id][visit_id]
                trajectory_list.append([visit_id, visit_time, obs])
            trajectory_list.sort(key=lambda x: x[0])
            trajectory_dict[patient_id] = trajectory_list

        for patient_id in obs_dict:
            trajectory_list = trajectory_dict[patient_id]
            single_hidden_state_list = viterbi_algorithm(trajectory_list, generate_mat, init_vec, obs_para_dict)
            hidden_state_dict[patient_id] = single_hidden_state_list
        return hidden_state_dict

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

    def __find_first_visit_id(self, patient_id):
        # find last visit
        first_visit = 100000
        first_time = 100000
        for visit_id in self.__visit_time_dict[patient_id]:
            visit_time = self.__visit_time_dict[patient_id][visit_id]
            if visit_time < first_time:
                first_time = visit_time
                first_visit = visit_id
        return first_visit

    def save_result(self, save_folder):
        result_dict = dict()
        result_dict['number of state'] = self.__num_state
        result_dict['alpha prior'] = self.__init_alpha
        result_dict['beta prior'] = self.__init_beta
        result_dict['phi prior'] = self.__init_phi
        result_dict['gamma prior'] = self.__init_gamma
        result_dict['iteration'] = self.__iteration
        result_dict['forward jump state range'] = self.__forward_candidate
        result_dict['backward jump state range'] = self.__backward_candidate
        result_dict['init state candidate'] = self.__init_state_candidate
        result_dict['parallel sampling time'] = self.__parallel_sampling_time
        result_dict['iteration'] = self.__iteration
        result_dict['reload parameter path'] = self.__reload_path

        with open(os.path.join(save_folder, 'result_parameter.csv'), "w",
                  encoding='utf-8-sig', newline='') as f:
            basic_data = [['parameter', 'value']]
            for key in result_dict:
                basic_data.append([key, result_dict[key]])
            csv.writer(f).writerows(basic_data)

        with open(os.path.join(save_folder, 'log_likelihood_trend.csv'), "w",
                  encoding='utf-8-sig', newline='') as f:
            basic_data = [['iteration', 'log_likelihood']]
            for item in self.__log_likelihood_trend:
                basic_data.append([item[0], item[1]])
            csv.writer(f).writerows(basic_data)

        sojourn_time_dict = self.__sojourn_time_dict
        transition_count_dict = self.__transition_count_dict
        observation_count_dict = self.__observation_count_dict
        init_state_dict = self.__init_state_dict

        # stat transition count
        data_to_write = list()
        for parallel_index in range(len(transition_count_dict)):
            data_to_write.append(['Parallel Index {}'.format(parallel_index + 1)])
            head = list()
            head.append('')
            for i in range(self.__num_state):
                head.append('state {}'.format(i + 1))
            data_to_write.append(head)

            for i, line in enumerate(transition_count_dict[parallel_index]):
                line_ = list()
                line_.append('state {}'.format(i + 1))
                for item in line:
                    line_.append(item)
                data_to_write.append(line_)
            data_to_write.append('')

        with open(os.path.join(save_folder, 'transition_count.csv'), "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # stat sojourn time
        with open(os.path.join(save_folder, 'sojourn_time.csv'), "w", encoding='utf-8-sig', newline='') as f:
            data_to_write = list()
            for parallel_index in range(len(sojourn_time_dict)):
                data_to_write.append(['Parallel Index {}'.format(parallel_index + 1)])
                head = list()
                head.append('')
                for i in range(self.__num_state):
                    head.append('state {}'.format(i + 1))
                data_to_write.append(head)

                sojourn_time_list = sojourn_time_dict[parallel_index]
                line_ = list()
                line_.append('')
                for item in sojourn_time_list:
                    line_.append(item)
                data_to_write.append(line_)
                data_to_write.append('')
            csv.writer(f).writerows(data_to_write)

        index_name_dict = self.__index_name_dict
        # stat observation count
        data_to_write = list()
        for parallel_index in observation_count_dict:
            data_to_write.append(['Parallel Index', parallel_index + 1])
            head_1 = list()
            head_1.append('Observation')
            head_1.append('Observation Type')
            for i in range(self.__num_state):
                head_1.append('')
                head_1.append('Mean')
                head_1.append('Variance(if exist)')
            data_to_write.append(head_1)
            head_2 = list()
            head_2.append('')
            head_2.append('')
            for i in range(self.__num_state):
                head_2.append('')
                head_2.append('State {}'.format(i+1))
                head_2.append('')
            data_to_write.append(head_2)
            for obs_index in observation_count_dict[parallel_index]:
                single_line = list()
                obs_name = index_name_dict[obs_index]
                obs_type = observation_count_dict[parallel_index][obs_index][0]
                obs_data = observation_count_dict[parallel_index][obs_index][1]
                single_line.append(obs_name)
                single_line.append(obs_type)
                for i in range(self.__num_state):
                    single_line.append('')
                    count = len(obs_data[i])
                    if count == 0:
                        single_line.append('None')
                        single_line.append('None')
                    else:
                        if obs_type == 'gaussian':
                            single_line.append(np.mean(obs_data[i]))
                            single_line.append(np.var(obs_data[i]))
                        elif obs_type == 'binomial':
                            single_line.append(np.mean(obs_data[i]))
                            single_line.append('None')
                        else:
                            raise ValueError('')
                data_to_write.append(single_line)
            data_to_write.append([])
            data_to_write.append([])
        with open(os.path.join(save_folder, 'observation_count.csv'), "w",
                  encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # init count
        data_to_write = list()
        head = list()
        head.append('')
        for i in range(self.__num_state):
            head.append('State {}'.format(i + 1))
        data_to_write.append(head)
        for parallel_index in init_state_dict:
            line = list()
            line.append('Parallel Index {}'.format(parallel_index + 1))
            for item in init_state_dict[parallel_index]:
                line.append(item)
            data_to_write.append(line)
        with open(os.path.join(save_folder, 'init_state_count.csv'),
                  "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # output estimated parameter
        generate_mat, init_vec, obs_para_dict = self.__estimate_parameter()
        # Generate Matrix
        data_to_write = list()
        head = list()
        head.append('')
        for i in range(self.__num_state):
            head.append('State {}'.format(i + 1))
        data_to_write.append(head)
        for i, item in enumerate(generate_mat):
            line = list()
            line.append('State {}'.format(i + 1))
            for state in item:
                line.append(state)
            data_to_write.append(line)
        with open(os.path.join(save_folder, 'estimated_generator_matrix.csv'),
                  "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # init vec
        data_to_write = list()
        head = list()
        for i in range(self.__num_state):
            head.append('State {}'.format(i + 1))
        data_to_write.append(head)
        line = list()
        for i in range(len(init_vec)):
            line.append(init_vec[i])
        data_to_write.append(line)
        with open(os.path.join(save_folder, 'estimated_init_state_distribution.csv'),
                  "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

        # obs para
        data_to_write = list()
        head_1 = list()
        head_1.append('Observation')
        head_1.append('Observation Type')
        for i in range(self.__num_state):
            head_1.append('')
            head_1.append('Posterior Mean')
            head_1.append('Preset Variance(if exist)')
        data_to_write.append(head_1)
        head_2 = list()
        head_2.append('')
        head_2.append('')
        for i in range(self.__num_state):
            head_2.append('')
            head_2.append('State {}'.format(i+1))
            head_2.append('')
        data_to_write.append(head_2)
        for obs_index in obs_para_dict:
            single_line = list()
            obs_name = index_name_dict[obs_index]
            obs_type = obs_para_dict[obs_index][0]['distribution']
            single_line.append(obs_name)
            single_line.append(obs_type)
            for i in range(self.__num_state):
                single_line.append('')
                if obs_type == 'gaussian':
                    single_line.append(obs_para_dict[obs_index][i]['sampled_mean'])
                    single_line.append(obs_para_dict[obs_index][i]['preset_variance'])
                elif obs_type == 'binomial':
                    single_line.append(obs_para_dict[obs_index][i]['prob'])
                    single_line.append('None')
                else:
                    raise ValueError('')
            data_to_write.append(single_line)
        with open(os.path.join(save_folder, 'estimated_observation_parameter.csv'),
                  "w", encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    def last_visit_state_prob(self, visit_time_dict):
        generate_mat, init_vec, obs_para_dict = self.__estimate_parameter()
        forward_procedure_dict = self.__inhomogeneous_forward_procedure(visit_time_dict, generate_mat, init_vec,
                                                                        obs_para_dict)

        last_visit_prob_state_dict = dict()
        for patient_id in forward_procedure_dict:
            log_last_visit_prob = np.array(forward_procedure_dict[patient_id][-1])
            minimum_prob = np.min(log_last_visit_prob)
            # rescale
            rescaled_log_prob = log_last_visit_prob - minimum_prob
            prob = np.exp(rescaled_log_prob) / np.sum(np.exp(rescaled_log_prob))
            last_visit_prob_state_dict[patient_id] = prob
        return last_visit_prob_state_dict


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
    """
    :param file_name:
    :param limitation: the allowed patient number
    :return:
    """
    obs_dict = dict()
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
            if not obs_dict.__contains__(patient_id):
                if limitation is not None:
                    if len(obs_dict) > limitation:
                        break
                obs_dict[patient_id] = dict()

            obs_dict[patient_id][visit_id] = list()
            for item in line[3:]:
                if item is None or len(item) == 0:
                    obs_dict[patient_id][visit_id].append(None)
                else:
                    obs_dict[patient_id][visit_id].append(float(item))
            if not visit_time_dict.__contains__(patient_id):
                visit_time_dict[patient_id] = dict()
            visit_time_dict[patient_id][visit_id] = visit_time
    return obs_dict, visit_time_dict, index_name_dict


def save_hidden_state_assignment(save_folder, state_assignment):
    data_to_write = list()
    head = ['patient_index', 'visit_index', 'estimated_state']
    data_to_write.append(head)
    for patient_index in state_assignment:
        state_list = state_assignment[patient_index]
        for visit_index, estimated_state in enumerate(state_list):
            data_to_write.append([patient_index, visit_index+1, estimated_state])
    with open(os.path.join(save_folder, 'hidden_state_assignment.csv'),
              "w", encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def save_mixture_of_last_visit(save_folder, last_visit_state_distribution, num_state):
    data_to_write = list()
    head = ['patient_id']
    for i in range(num_state):
        head.append('state {}'.format(i+1))
    data_to_write.append(head)
    for patient_id in last_visit_state_distribution:
        line = [patient_id]
        for item in last_visit_state_distribution[patient_id]:
            line.append(item)
        data_to_write.append(line)
    with open(os.path.join(save_folder, 'mixture_of_last_visit.csv'),
              "w", encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def unit_test():
    optimization_time = 30
    likelihood_calculate_interval = 10
    limitation = None
    num_state = 4
    alpha = [1, 0.001]
    beta = 0.1
    gamma_file_name = os.path.abspath('../../resource/特征先验_混合.csv')
    data_file_name = \
        os.path.abspath('../../resource/未预处理长期纵向数据_离散化_False_特别筛选_True_截断年份_2009.csv')
    gamma_dict = read_prior(gamma_file_name)
    phi = 0.1
    parallel_sampling_time = 1
    forward_candidate = 1
    backward_candidate = 0
    init_state_candidate = 3

    obs_dict, visit_time_dict, index_name_dict = read_data(data_file_name, limitation=limitation)

    cdpm = ContinuousDiseaseProgressModel(obs_dict, visit_time_dict, index_name_dict,
                                          parallel_sampling_time=parallel_sampling_time, num_state=num_state,
                                          alpha=alpha, beta=beta, gamma=gamma_dict, phi=phi,
                                          forward_candidate=forward_candidate, backward_candidate=backward_candidate,
                                          init_state_candidate=init_state_candidate)

    cdpm.optimization(optimization_time, likelihood_calculate_interval=likelihood_calculate_interval)
    state_assignment = cdpm.disease_state_assignment(obs_dict=obs_dict, time_dict=visit_time_dict)

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_folder = os.path.abspath('../../resource/result/CDPM/{}'.format(now))
    os.mkdir(save_folder)
    cdpm.save_result(save_folder)
    save_hidden_state_assignment(save_folder, state_assignment)
    last_visit_state_distribution = cdpm.last_visit_state_prob(visit_time_dict=visit_time_dict)
    save_mixture_of_last_visit(save_folder, last_visit_state_distribution, num_state)


if __name__ == '__main__':
    unit_test()
