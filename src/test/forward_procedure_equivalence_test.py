import numpy as np
import random
import math
"""
Conclusion
Forward Procedure 1 will meet numerical unstable problem when calculating the likelihood in long sequence
Forward Procedure 2 is wrong, where I don't know the reason. But the error is tolerable (typically less than 0.5%)
Forward Procedure 3 is numerical stable
"""


def main():
    num_state = 5
    num_observation = 50

    transition_matrix = np.random.uniform(0, 1, [num_state, num_state])
    observation_matrix = np.random.uniform(0, 1, [num_state, num_observation])
    for i in range(len(transition_matrix)):
        row_sum = np.sum(transition_matrix[i])
        transition_matrix[i] = transition_matrix[i] / row_sum
    for i in range(len(observation_matrix)):
        row_sum = np.sum(observation_matrix[i])
        observation_matrix[i] = observation_matrix[i] / row_sum
    print('initialized')

    corpus = list()
    for i in range(40):
        corpus.append(list())
        for j in range(20):
            corpus[-1].append(random.randint(0, num_observation - 1))

    for i in range(len(corpus)):
        log_prob_1 = forward_procedure_1(corpus[i], transition_matrix, observation_matrix)
        log_prob_2 = forward_procedure_2(corpus[i], transition_matrix, observation_matrix)
        log_prob_3 = forward_procedure_3(corpus[i], transition_matrix, observation_matrix)
        print('prob_1: {:.10f}, prob_2: {:.10f}, prob_3: {:.10f}. prob_1/prob_2: {:.10f}, prob_2/prob_3: {:.10f},'
              ' prob_1/prob_3: {:.10f}'
              .format(log_prob_1, log_prob_2, log_prob_3, log_prob_1/log_prob_2, log_prob_2/log_prob_3,
                      log_prob_1/log_prob_3))


def forward_procedure_1(doc, transition_matrix, observation_matrix):
    cache = list()
    cache.append([1/len(transition_matrix) for _ in range(len(transition_matrix))])
    for i in range(len(doc)):
        slot = list()
        for j in range(len(transition_matrix)):
            prob_j = 0
            for k in range(len(transition_matrix)):
                prob_j += cache[-1][k] * transition_matrix[k][j] * observation_matrix[j][doc[i]]
            slot.append(prob_j)
        cache.append(slot)
    doc_prob = 0
    for item in cache[-1]:
        doc_prob += item
    doc_prob = math.log(doc_prob)
    return doc_prob


# Wrong, but the error is acceptable
def forward_procedure_2(doc, transition_matrix, observation_matrix):
    cache = list()
    cache.append([1 / len(transition_matrix) for _ in range(len(transition_matrix))])
    for i in range(len(doc)):
        slot = list()
        for j in range(len(transition_matrix)):
            prob_j = 0
            for k in range(len(transition_matrix)):
                prob_j += cache[-1][k] * transition_matrix[k][j]
            slot.append(prob_j)
        cache.append(slot)

    doc_prob = 0
    for i in range(len(doc)):
        token_prob = 0
        for j in range(len(transition_matrix)):
            token_prob += cache[i+1][j] * observation_matrix[j][doc[i]]
        doc_prob += math.log(token_prob)
    return doc_prob


def forward_procedure_3(doc, transition_matrix, observation_matrix):
    cache = list()
    cache.append([math.log(1/len(transition_matrix)) for _ in range(len(transition_matrix))])
    for i in range(len(doc)):
        slot = list()
        for j in range(len(transition_matrix)):
            prob_j = 0
            for k in range(len(transition_matrix)):
                eln_prod = cache[-1][k] + math.log(transition_matrix[k][j])
                if k == 0:
                    prob_j = eln_prod
                else:
                    if prob_j > eln_prod:
                        prob_j = prob_j + math.log(1 + math.exp(eln_prod - prob_j))
                    else:
                        prob_j = eln_prod + math.log(1 + math.exp(prob_j - eln_prod))
            prob_j += math.log(observation_matrix[j][doc[i]])
            slot.append(prob_j)
        cache.append(slot)

    # calculate the final probability
    doc_prob = 0
    for k in range(len(cache[-1])):
        log_sum_k = cache[-1][k]
        if k == 0:
            doc_prob = log_sum_k
        else:
            if doc_prob > log_sum_k:
                doc_prob = doc_prob + math.log(1 + math.exp(log_sum_k - doc_prob))
            else:
                doc_prob = log_sum_k + math.log(1 + math.exp(doc_prob - log_sum_k))
    return doc_prob


if __name__ == '__main__':
    main()

