# encoding=utf-8
import os
import datetime
import data_reader
import numpy as np
from disease_progress_model import DiseaseProgressionModel, save_risk_tier_assignment, save_disease_state_assignment


def main():
    file_path = os.path.abspath('../../resource/二值化后的长期纵向数据.csv')
    document, index_name_dict, index_patient_dict = data_reader.data_reader(file_path)

    obs_num = 107
    iteration = 2400
    patient_subtype_list = [6]
    risk_tier = 2
    alpha = 1
    beta = np.full(obs_num, 0.1)
    gamma = 0.1
    update_interval = 100

    for patient_subtype in patient_subtype_list:
        dependence_constraint = init_state_candidate = patient_subtype // 2
        disease_progression_model = DiseaseProgressionModel(document, alpha=alpha, beta=beta, gamma=gamma,
                                                            num_risk_tier=risk_tier, num_subtype=patient_subtype,
                                                            dependence_constraint=dependence_constraint,
                                                            init_state_candidate=init_state_candidate)
        disease_progression_model.optimization(iteration, update_interval=update_interval)

        now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        model_save_path = os.path.abspath('../../resource/result/DPM/{}'.format(now))
        os.mkdir(model_save_path)
        os.mkdir(os.path.join(model_save_path, 'inference_result'))
        inference_path = os.path.join(model_save_path, 'inference_result')

        disease_progression_model.save_result(model_save_path, index_name_dict, index_patient_dict)

        estimated_risk_list = disease_progression_model.risk_tier_inference(document)
        estimated_state_list = disease_progression_model.disease_state_assignment(document)
        save_disease_state_assignment(estimated_state_list, index_patient_dict, inference_path)
        save_risk_tier_assignment(estimated_risk_list, index_patient_dict, inference_path)


if __name__ == '__main__':
    main()
