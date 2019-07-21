# encoding=utf-8
import os
import csv


def data_reader(file_path):
    index_name_dict = dict()
    index_patient_dict = dict()
    medical_data = dict()
    with open(file_path, 'r', encoding='gbk', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                head_flag = False
                # 注意，由于用的是离散HMM，不考虑时间差的影响
                for i in range(len(line)-3):
                    index_name_dict[i] = line[i+2]
                continue
            patient_id = line[0]
            if not medical_data.__contains__(patient_id):
                medical_data[patient_id] = list()

            single_visit = list()
            for index, item in enumerate(line[2: -1]):
                if item == '1':
                    single_visit.append(index)
            medical_data[patient_id].append(single_visit)

    dict_to_list = list()
    for i, patient_id in enumerate(medical_data):
        patient_list = list()
        index_patient_dict[i] = patient_id
        for visit_list in medical_data[patient_id]:
            patient_list.append(visit_list)
        dict_to_list.append(patient_list)
    return dict_to_list, index_name_dict, index_patient_dict


if __name__ == "__main__":
    path = os.path.abspath('../../resource/二值化后的长期纵向数据.csv')
    print(path)
    data_reader(path)
    print('accomplish')
