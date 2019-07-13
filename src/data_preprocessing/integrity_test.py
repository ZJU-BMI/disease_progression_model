# 本脚本用于计算目标特征的起算时间和目标数据中的完整度
import csv
import os
import math
import numpy as np
import re 
from itertools import islice
import datetime

def read_candidate_pat_visit(year):
    legal_patient_visit = dict()
    pat_visit_list = list()
    with open(os.path.abspath('resource/data/pat_visit.csv'), 'r', encoding='gbk', newline='') as pat_visit_file:
        csv_reader = csv.reader(pat_visit_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, admission_time = line[0], line[1], line[3]
            if len(admission_time) < 6:
                continue
            elif len(admission_time) < 12:
                happen_time = datetime.datetime.strptime(admission_time, '%Y/%m/%d')
            else:
                happen_time = datetime.datetime.strptime(admission_time, '%Y/%m/%d %H:%M:%S')
            admit_year = happen_time.year
            if admit_year < year:
                continue
        
            if not legal_patient_visit.__contains__(patient_id):
                legal_patient_visit[patient_id] = []
        
            legal_patient_visit[patient_id].append(visit_id)
            

    with open(os.path.abspath('resource/候选访问.csv'), 'r', encoding='gbk', newline='') as candiate_pat_visit:
        csv_reader = csv.reader(candiate_pat_visit)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id = line[0], line[1]
            if legal_patient_visit.__contains__(patient_id):
                visit_list = legal_patient_visit[patient_id]
                if visit_list.__contains__(visit_id):
                    pat_visit_list.append([patient_id, visit_id])

    patient_set = set()
    for item in pat_visit_list:
        patient_set.add(item[0])
    print('year: {}, patient: {}, visit: {}'.format(year, len(patient_set), len(pat_visit_list)))

    return pat_visit_list

def read_lab_test_start_time(item_dict):
    item_time_dict = dict()
    for key in item_dict:
        item_time_dict[item_dict[key]] = list()

    with open(os.path.abspath('resource/data/lab_result.csv'), 'r', encoding='gbk', newline='') as candiate_pat_visit:
        csv_reader = csv.reader(candiate_pat_visit)
        for line in islice(csv_reader, 1, None):
            item_name, test_time = line[3], line[9]
            if item_dict.__contains__(item_name):
                if len(test_time) < 6:
                    continue
                elif len(test_time) < 12:
                    happen_time = datetime.datetime.strptime(test_time, '%Y/%m/%d')
                else:
                    happen_time = datetime.datetime.strptime(test_time, '%Y/%m/%d %H:%M:%S')
                item_time_dict[item_dict[item_name]].append(happen_time)
    
    item_start_time_dict = dict()
    for key in item_time_dict:
        item_time_dict[key] = sorted(item_time_dict[key])
        cut_point = math.floor(len(item_time_dict[key])*0.05)
        item_start_time_dict[key] = item_time_dict[key][cut_point]
    return item_start_time_dict

def lab_test_integrity(pat_visit_list, labtest_dict):
    lab_test_result_dict = dict()
    with open(os.path.abspath('resource/data/lab_result.csv'), 'r', encoding='gbk', newline='') as lab_test_file:
        csv_reader = csv.reader(lab_test_file)
        for line in islice(csv_reader, 1, None):
            test_no, item_no, item_name = line[0], line[1], line[3]
            if labtest_dict.__contains__(item_name):
                if not lab_test_result_dict.__contains__(test_no):
                    lab_test_result_dict[test_no] = dict()
                lab_test_result_dict[test_no][item_no] = item_name
    
    # data template
    pat_labtest_dict = dict()
    for item in pat_visit_list:
        if not pat_labtest_dict.__contains__(item[0]):
            pat_labtest_dict[item[0]] = dict()
        pat_labtest_dict[item[0]][item[1]] = dict()
        for key in labtest_dict:
            pat_labtest_dict[item[0]][item[1]][labtest_dict[key]] = False
    
    with open(os.path.abspath('resource/data/lab_test_master.csv'), 'r', encoding='gbk', newline='') as lab_master_file:
        csv_reader = csv.reader(lab_master_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, test_no = line[2], line[3], line[0]
            if pat_labtest_dict.__contains__(patient_id) and \
                pat_labtest_dict[patient_id].__contains__(visit_id) and \
                    lab_test_result_dict.__contains__(test_no):
                for lab_test_item in lab_test_result_dict[test_no]:
                    lab_test_name = lab_test_result_dict[test_no][lab_test_item]
                    if not labtest_dict.__contains__(lab_test_name):
                        continue
                    normalized_name = labtest_dict[lab_test_name]
                    pat_labtest_dict[patient_id][visit_id][normalized_name] = True

    count_dict = dict()
    for key in labtest_dict:
        count_dict[labtest_dict[key]] = 0
    for patient_id in pat_labtest_dict:
        for visit_id in pat_labtest_dict[patient_id]:
            for item_name in pat_labtest_dict[patient_id][visit_id]:
                show_flag = pat_labtest_dict[patient_id][visit_id][item_name]
                if show_flag:
                    count_dict[item_name] += 1
    for key in count_dict:
        count_dict[key] = count_dict[key] / len(pat_visit_list)
    for key in count_dict:
        print('{}: {}'.format(key, count_dict[key]))


def ecg_integrity(pat_visit_list):
    # data template
    pat_ecg_dict = dict()
    for item in pat_visit_list:
        if not pat_ecg_dict.__contains__(item[0]):
            pat_ecg_dict[item[0]] = dict()
        pat_ecg_dict[item[0]][item[1]] = False
    
    valid_exam_no_dict= dict()
    with open(os.path.abspath('resource/data/exam_master.csv'), 'r', encoding='gbk', newline='') as exam_master_file:
        csv_reader = csv.reader(exam_master_file)
        for line in islice(csv_reader, 1, None):
            exam_no, patient_id, visit_id, exam_class = line[0], line[3], line[35], line[7]
            if pat_ecg_dict.__contains__(patient_id) and pat_ecg_dict[patient_id].__contains__(visit_id):
                if exam_class == '心电图':
                    valid_exam_no_dict[exam_no] = [patient_id, visit_id]
    with open(os.path.abspath('resource/data/exam_report.csv'), 'r', encoding='gb18030', newline='') as exam_report_file:
        csv_reader = csv.reader(exam_report_file)
        for line in islice(csv_reader, 1, None):
            exam_no, description = line[0], line[3]
            if valid_exam_no_dict.__contains__(exam_no) and len(description) > 4:
                patient_id, visit_id = valid_exam_no_dict[exam_no]
                pat_ecg_dict[patient_id][visit_id] = True
    
    total_count = 0
    positive_count = 0
    for patient_id in pat_ecg_dict:
        for visit_id in pat_ecg_dict[patient_id]:
            total_count += 1
            if pat_ecg_dict[patient_id][visit_id]:
                positive_count += 1
    print('ECG Integrity is: {}'.format(positive_count/total_count))


def echocardiogram_integrity(pat_visit_list):
    # data template
    pat_echocardiogram_dict = dict()
    for item in pat_visit_list:
        if not pat_echocardiogram_dict.__contains__(item[0]):
            pat_echocardiogram_dict[item[0]] = dict()
        pat_echocardiogram_dict[item[0]][item[1]] = False
    
    valid_exam_no_dict= dict()
    with open(os.path.abspath('resource/data/exam_master.csv'), 'r', encoding='gbk', newline='') as exam_master_file:
        csv_reader = csv.reader(exam_master_file)
        for line in islice(csv_reader, 1, None):
            exam_no, patient_id, visit_id, exam_class, exam_sub_class = line[0], line[3], line[35], line[7], line[8]
            if pat_echocardiogram_dict.__contains__(patient_id) and pat_echocardiogram_dict[patient_id].__contains__(visit_id):
                if exam_class == '超声' and (exam_sub_class == '心脏' or exam_sub_class == '床旁超声'):
                    valid_exam_no_dict[exam_no] = [patient_id, visit_id]
    with open(os.path.abspath('resource/data/exam_report.csv'), 'r', encoding='gb18030', newline='') as exam_report_file:
        csv_reader = csv.reader(exam_report_file)
        for line in islice(csv_reader, 1, None):
            exam_no, exam_para = line[0], line[1]
            if valid_exam_no_dict.__contains__(exam_no) and len(exam_para) > 40:
                patient_id, visit_id = valid_exam_no_dict[exam_no]
                pat_echocardiogram_dict[patient_id][visit_id] = True
    
    total_count = 0
    positive_count = 0
    for patient_id in pat_echocardiogram_dict:
        for visit_id in pat_echocardiogram_dict[patient_id]:
            total_count += 1
            if pat_echocardiogram_dict[patient_id][visit_id]:
                positive_count += 1
    print('echocardiogram Integrity is: {}'.format(positive_count/total_count))


def cardio_ct_integrity(pat_visit_list):
    # data template
    pat_ct_dict = dict()
    for item in pat_visit_list:
        if not pat_ct_dict.__contains__(item[0]):
            pat_ct_dict[item[0]] = dict()
        pat_ct_dict[item[0]][item[1]] = False
    
    valid_exam_no_dict= dict()
    with open(os.path.abspath('resource/data/exam_master.csv'), 'r', encoding='gbk', newline='') as exam_master_file:
        csv_reader = csv.reader(exam_master_file)
        for line in islice(csv_reader, 1, None):
            exam_no, patient_id, visit_id, exam_class, exam_sub_class = line[0], line[3], line[35], line[7], line[8]
            if pat_ct_dict.__contains__(patient_id) and pat_ct_dict[patient_id].__contains__(visit_id):
                if exam_class == 'ＣＴ' and (exam_sub_class == '血管心脏' or exam_sub_class == '胸部'):
                    valid_exam_no_dict[exam_no] = [patient_id, visit_id]
    with open(os.path.abspath('resource/data/exam_report.csv'), 'r', encoding='gb18030', newline='') as exam_report_file:
        csv_reader = csv.reader(exam_report_file)
        for line in islice(csv_reader, 1, None):
            exam_no, exam_para, description, impression = line[0], line[1], line[2], line[3]
            if valid_exam_no_dict.__contains__(exam_no) and (len(exam_para) + len(description) + len(impression) > 10):
                patient_id, visit_id = valid_exam_no_dict[exam_no]
                pat_ct_dict[patient_id][visit_id] = True
    
    total_count = 0
    positive_count = 0
    for patient_id in pat_ct_dict:
        for visit_id in pat_ct_dict[patient_id]:
            total_count += 1
            if pat_ct_dict[patient_id][visit_id]:
                positive_count += 1
    print('CT Integrity is: {}'.format(positive_count/total_count))


def lab_test_distribution(item_dict):
    item_distribution = dict()
    for key in item_dict:
        item_distribution[item_dict[key]] = list()
    with open(os.path.abspath('resource/data/lab_result.csv'), 'r', encoding='gbk', newline='') as lab_test_file:
        csv_reader = csv.reader(lab_test_file)
        for line in islice(csv_reader, 1, None):
            item_name, item_value = line[3], line[5]
            item_value = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', item_value)
            if len(item_value) == 0:
                continue
            try:
                item_value = float(item_value[0])
                if item_dict.__contains__(item_name):
                    item_distribution[item_dict[item_name]].append(float(item_value))
            except:
                continue
    with open(os.path.abspath('resource/lab_result_distribution.csv'), 'w', encoding='gbk', newline='') as lab_test_file:
        max_line = 0
        for key in item_distribution:
            if max_line < len(item_distribution[key]):
                max_line = len(item_distribution[key])
        result = np.zeros([len(item_distribution), max_line])
    
        index = 0
        name_list = list()
        for key in item_distribution:
            name_list.append(key)
            for j, item in enumerate(item_distribution[key]):
                result[index, j] = item
            index += 1
        result = result.transpose()
        csv.writer(lab_test_file).writerows([name_list])
        csv.writer(lab_test_file).writerows(result)

def main():
     
    item_dict = {'脑利钠肽前体': '脑利钠肽前体', '肌钙蛋白T': '肌钙蛋白T', '钾': '钾', '钠': '钠', '肌酐': '肌酐',
    '血清促甲状腺激素测定': '血清促甲状腺激素测定',  '葡萄糖': '葡萄糖', '全血糖化血红蛋白测定': '全血糖化血红蛋白测定',
    '降钙素原': '降钙素原', '降钙素原（发光法）': '降钙素原', '白细胞介素-6': '白细胞介素-6', 'C-反应蛋白测定': 'C-反应蛋白测定', 
    '低密度脂蛋白胆固醇': '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇': '高密度脂蛋白胆固醇', '总胆固醇': '总胆固醇',
    '甘油三酯': '甘油三酯', '血清白蛋白': '白蛋白', '白蛋白': '白蛋白', '血浆纤维蛋白原测定': '血浆纤维蛋白原测定', '铁': '铁',
    '红细胞比积测定': '红细胞比积测定', '血清铁蛋白': '血清铁蛋白', '总铁结合力': '总铁结合力', '血浆D-二聚体测定': '血浆D-二聚体测定' ,
    '血小板体积分布宽度': '血小板体积分布宽度', '血浆凝血酶原活动度测定': '血浆凝血酶原活动度测定', '红细胞沉降率测定': '红细胞沉降率测定',
    '血浆凝血酶原时间测定': '血浆凝血酶原时间测定'}
    # item_start_time_dict = read_lab_test_start_time(item_dict)
    # for key in item_start_time_dict:
    #     print('{}:  {}'.format(key, item_start_time_dict[key]))
    # print('accomplish')

    # patient_list = read_candidate_pat_visit(year = 2005)
    # lab_test_integrity(patient_list, item_dict)
    # ecg_integrity(patient_list)
    # echocardiogram_integrity(patient_list)
    # cardio_ct_integrity(patient_list)
    lab_test_distribution(item_dict)
    print('accomplished')


if __name__ == '__main__':
    main()
