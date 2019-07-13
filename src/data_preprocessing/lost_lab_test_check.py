import os
import csv
from itertools import islice
import integrity_test

# 本脚本拟探究labtest的缺失到底是不是因为部门差异导致的
def main():
    pat_visit_list = integrity_test.read_candidate_pat_visit(2010)
    pat_visit_list = heart_failure_filter(pat_visit_list)
    pat_visit_dept_dict = get_patient_admission_dept(pat_visit_list)
    pat_labtest_dict = get_pat_visit_has_bnp_record(pat_visit_list)
    dept_integrity_test(pat_labtest_dict, pat_visit_dept_dict)

def heart_failure_filter(pat_visit_list):
    pat_filtered_list = list()
    pat_valid_dict = dict()
    with open(os.path.abspath('resource/data/diagnosis.csv'), 'r', encoding='gbk', newline='') as pat_visit_file:
        csv_reader = csv.reader(pat_visit_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, diagnosis_code, diagnosis = line[0], line[1], line[2], line[4]
            if diagnosis_code == '3' or diagnosis_code == 'A':
                if diagnosis.__contains__('心功能') or diagnosis.__contains__('心力衰竭') or diagnosis.__contains__('心衰'):
                    if not pat_valid_dict.__contains__(patient_id):
                        pat_valid_dict[patient_id] = set()
                    pat_valid_dict[patient_id].add(visit_id)
    for item in pat_visit_list:
        patient_id, visit_id = item
        if pat_valid_dict.__contains__(patient_id) and pat_valid_dict[patient_id].__contains__(visit_id):
            pat_filtered_list.append([patient_id, visit_id])
    return pat_filtered_list

def get_patient_admission_dept(pat_visit_list):
    pat_visit_dict = dict()
    pat_visit_dept_dict = dict()

    for item in pat_visit_list:
        patient_id, visit_id = item
        if not pat_visit_dict.__contains__(patient_id):
            pat_visit_dict[patient_id] = set()
        pat_visit_dict[patient_id].add(visit_id)

    with open(os.path.abspath('resource/data/pat_visit.csv'), 'r', encoding='gbk', newline='') as pat_visit_file:
        csv_reader = csv.reader(pat_visit_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, admission_dept = line[0], line[1], line[2]
            if pat_visit_dict.__contains__(patient_id) and pat_visit_dict[patient_id].__contains__(visit_id):
                if not pat_visit_dept_dict.__contains__(patient_id):
                    pat_visit_dept_dict[patient_id] = dict()
                pat_visit_dept_dict[patient_id][visit_id] = admission_dept
    return pat_visit_dept_dict

def get_pat_visit_has_bnp_record(pat_visit_list):
    lab_test_result_set = set()
    with open(os.path.abspath('resource/data/lab_result.csv'), 'r', encoding='gbk', newline='') as lab_test_file:
        csv_reader = csv.reader(lab_test_file)
        for line in islice(csv_reader, 1, None):
            test_no, item_name = line[0], line[3]
            if item_name == '脑利钠肽前体':
                lab_test_result_set.add(test_no)
    
    # data template
    pat_labtest_dict = dict()
    for item in pat_visit_list:
        if not pat_labtest_dict.__contains__(item[0]):
            pat_labtest_dict[item[0]] = dict()
        pat_labtest_dict[item[0]][item[1]] = False
    
    with open(os.path.abspath('resource/data/lab_test_master.csv'), 'r', encoding='gbk', newline='') as lab_master_file:
        csv_reader = csv.reader(lab_master_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, test_no = line[2], line[3], line[0]
            if pat_labtest_dict.__contains__(patient_id) and \
                pat_labtest_dict[patient_id].__contains__(visit_id) and \
                    lab_test_result_set.__contains__(test_no):
                    pat_labtest_dict[patient_id][visit_id] = True
    return pat_labtest_dict


def dept_integrity_test(pat_labtest_dict, pat_visit_dept_dict):
    dept_count_dict = dict()
    dept_positive_dict = dict()
    for patient_id in pat_visit_dept_dict:
        for visit_id in pat_visit_dept_dict[patient_id]:
            dept_number = pat_visit_dept_dict[patient_id][visit_id]
            dept_count_dict[dept_number] = 0
            dept_positive_dict[dept_number] = 0
    for patient_id in pat_labtest_dict:
        for visit_id in pat_labtest_dict[patient_id]:
            bnp_exist = pat_labtest_dict[patient_id][visit_id]
            dept = pat_visit_dept_dict[patient_id][visit_id]

            if bnp_exist:
                dept_positive_dict[dept] += 1
            dept_count_dict[dept] += 1
    data_to_write = []
    for dept_name in dept_count_dict:
        print('{}: count: {}, integrity is: {}'.format(dept_name, 
        dept_count_dict[dept_name], 
        dept_positive_dict[dept_name]/dept_count_dict[dept_name]))
        data_to_write.append([dept_name, 
        dept_count_dict[dept_name], 
        dept_positive_dict[dept_name]/dept_count_dict[dept_name]])
    with open(os.path.abspath('resource/bnp_dept_integrity_test_2.csv'), 'w', encoding='gbk', newline='') as csv_file:
        csv.writer(csv_file).writerows(data_to_write)


if __name__ == '__main__':
    main()