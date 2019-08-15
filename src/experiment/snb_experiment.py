# encoding=utf-8
import os
import numpy as np
import datetime
import data_reader
from sequential_naive_bayes import SequentialNaiveBayes, save_disease_state_assignment


def main():
    file_path = os.path.abspath('../../resource/二值化后的长期纵向数据.csv')
    document, index_name_dict, index_patient_dict = data_reader.data_reader(file_path)

    obs_num = 107
    iteration = 4000
    dependence_constraint = 2
    patient_subtype = 4
    alpha = 1
    beta = np.full(obs_num, 0.01)
    update_interval = 100
    init_state_candidate = 3

    sequence_naive_bayes = SequentialNaiveBayes(document, alpha=alpha, beta=beta,
                                                num_subtype=patient_subtype, init_state_candidate=init_state_candidate,
                                                dependence_constraint=dependence_constraint)
    sequence_naive_bayes.optimization(iteration, update_interval=update_interval)
    estimated_state_list = sequence_naive_bayes.disease_state_assignment(document)

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_save_path = os.path.abspath('../../resource/result/SNB/{}'.format(now))
    os.mkdir(model_save_path)
    os.mkdir(os.path.join(model_save_path, 'inference_result'))
    inference_path = os.path.join(model_save_path, 'inference_result')

    sequence_naive_bayes.save_result(model_save_path, index_name_dict, index_patient_dict)
    save_disease_state_assignment(estimated_state_list, index_patient_dict, inference_path)


if __name__ == '__main__':
    main()
