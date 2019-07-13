# 负责从数据集中获取目标数据
import csv
import os
import re
from itertools import islice
import datetime


def get_valid_visit(data_path_dict, cache_root, read_from_cache=False, truncate_year=2009):
    """
    条件
    1.入院时间自2009年1月1日起算
    2.能够检验中找到能够关联到的项目
    3.能够查到有效的人口学数据
    4.能够查到有效的Order数据
    5.能够找到诊断数据
    6.经过上述所有筛选后，入院的入院次数大于3次

    :param data_path_dict:
    :param cache_root:
    :param read_from_cache:
    :return: 一个<Patient_ID, [Visit_ID, ..., Visit_ID]>结构的有效表格
    """

    if read_from_cache:
        visit_dict = dict()
        read_path = os.path.join(cache_root, 'visit_dict.csv')
        with open(read_path, 'r', encoding="gbk", newline="") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id = line[0]
                visit_id = line[1]
                if not visit_dict.__contains__(patient_id):
                    visit_dict[patient_id] = set()
                visit_dict[patient_id].add(visit_id)
        return visit_dict

    visit_path = data_path_dict['pat_visit']
    visit_dict = dict()
    with open(visit_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            admission_year = line[3][0:4]

            if int(admission_year) < truncate_year:
                continue

            if not visit_dict.__contains__(patient_id):
                visit_dict[patient_id] = set()
            visit_dict[patient_id].add(visit_id)

    # 删掉没有对应任何检验信息的病人
    lab_result_path = data_path_dict['lab_result']
    valid_test_no_set = set()
    with open(lab_result_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            valid_test_no_set.add(line[0])

    lab_test_path = data_path_dict['lab_test_master']
    lab_test_dict = dict()
    with open(lab_test_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            test_no = line[0]
            patient_id = line[2]
            visit_id = line[3]
            if not valid_test_no_set.__contains__(test_no):
                continue
            if not lab_test_dict.__contains__(patient_id):
                lab_test_dict[patient_id] = set()
            lab_test_dict[patient_id].add(visit_id)
            
    lab_test_invalid_list = list()
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            if not (lab_test_dict.__contains__(patient_id) and lab_test_dict[patient_id].__contains__(visit_id)):
                lab_test_invalid_list.append([patient_id, visit_id])
    for item in lab_test_invalid_list:
        patient_id, visit_id = item
        visit_dict[patient_id].discard(visit_id)

    # 能调查到人口学信息的病人ID
    patient_master_path = data_path_dict['pat_master_index']
    demographic_info_available_set = set()
    with open(patient_master_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            sex = line[4]
            birthday = line[5]
            if not (sex == ''  or len(birthday) < 6):
                demographic_info_available_set.add(patient_id)
    # 删减掉没有人口学信息的病人
    demographic_info_unavailable_set = set()
    for patient_id in visit_dict:
        if not demographic_info_available_set.__contains__(patient_id):
            demographic_info_unavailable_set.add(patient_id)
    for patient_id in demographic_info_unavailable_set:
        visit_dict.pop(patient_id)

    # 删掉没有Order数据的病人
    order_path = data_path_dict['orders']
    order_valid_dict = dict()
    with open(order_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            if not order_valid_dict.__contains__(patient_id):
                order_valid_dict[patient_id] = set()
            order_valid_dict[patient_id].add(visit_id)
    order_invalid_list = list()
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            if not (order_valid_dict.__contains__(patient_id) and order_valid_dict[patient_id].__contains__(visit_id)):
                order_invalid_list.append([patient_id, visit_id])
    for item in order_invalid_list:
        patient_id, visit_id = item
        visit_dict[patient_id].discard(visit_id)

    # 删除没有主诊断或其它诊断信息的数据
    diagnosis_valid_dict = dict()
    diagnosis_path = data_path_dict['diagnosis']
    with open(diagnosis_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            diagnosis_type = line[2]
            if not (diagnosis_type == 'A' or diagnosis_type == '3'):
                continue
            if not diagnosis_valid_dict.__contains__(patient_id):
                diagnosis_valid_dict[patient_id] = set()
            diagnosis_valid_dict[patient_id].add(visit_id)
    diagnosis_invalid_list = list()
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            if not (diagnosis_valid_dict.__contains__(patient_id) and diagnosis_valid_dict[patient_id].__contains__(
                    visit_id)):
                diagnosis_invalid_list.append([patient_id, visit_id])
    for item in diagnosis_invalid_list:
        patient_id, visit_id = item
        visit_dict[patient_id].discard(visit_id)

    # 删除入院次数小于4次的病人
    eliminate_patient = set()
    for patient_id in visit_dict:
        if len(visit_dict[patient_id]) < 3:
            eliminate_patient.add(patient_id)
    for item in eliminate_patient:
        visit_dict.pop(item)

    # 写数据
    data_to_write = list()
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            data_to_write.append([patient_id, visit_id])
    write_path = os.path.join(cache_root, 'visit_dict.csv')
    with open(write_path, 'w', encoding="gbk", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data_to_write)

    return visit_dict

def get_visit_info(visit_dict, data_path_dict, cache_root, read_from_cache=False):
    """
    给出每个病人每次入院的距离第一次入院的时间差，住院天数，是否死亡四个信息
    :param visit_dict:
    :param data_path_dict:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        visit_info_dict = dict()
        cache_path = os.path.join(cache_root, 'visit_info.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, item, value = line
                if not visit_info_dict.__contains__(patient_id):
                    visit_info_dict[patient_id] = dict()
                if not visit_info_dict[patient_id].__contains__(visit_id):
                    visit_info_dict[patient_id][visit_id] = dict()
                if item == '入院时间':
                    visit_info_dict[patient_id][visit_id][item] = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                else:
                    visit_info_dict[patient_id][visit_id][item] = value
        return visit_info_dict

    # 填充数据模板
    visit_info_dict = dict()
    for patient_id in visit_dict:
        visit_info_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            # 时间差，是否心源性入院，住院天数，是否死亡。其中，入院时间最终是要抹去的
            visit_info_dict[patient_id][visit_id] = {'时间差': -1, '入院时间': -1, '住院天数': -1, '死亡': 0}

    # 判断病人是否死亡
    diagnosis_path = data_path_dict['diagnosis']
    with open(diagnosis_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, _, _, _, _, _, result, _, _ = line
            if not (visit_info_dict.__contains__(patient_id) and visit_info_dict[patient_id].__contains__(visit_id)):
                continue
            if result == '死亡':
                visit_info_dict[patient_id][visit_id]['死亡'] = 1

    # 判断病人的入院时期和住院时间
    visit_path = data_path_dict['pat_visit']
    with open(visit_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            admission_date = line[3]
            discharge_date = line[5]
            if not (visit_info_dict.__contains__(patient_id) and visit_info_dict[patient_id].__contains__(visit_id)):
                continue
            if len(admission_date) > 12:
                admission_date = datetime.datetime.strptime(admission_date, '%Y/%m/%d %H:%M:%S')
            elif 4 < len(admission_date) < 12:
                admission_date = datetime.datetime.strptime(admission_date, '%Y/%m/%d')
            else:
                continue
            if len(discharge_date) > 12:
                discharge_date = datetime.datetime.strptime(discharge_date, '%Y/%m/%d %H:%M:%S')
            elif 4 < len(discharge_date) < 12:
                discharge_date = datetime.datetime.strptime(discharge_date, '%Y/%m/%d')
            else:
                continue
            time_interval = (discharge_date - admission_date).days
            visit_info_dict[patient_id][visit_id]['入院时间'] = admission_date
            visit_info_dict[patient_id][visit_id]['住院天数'] = time_interval

    # 判断病人距离第一次入院的时间差
    # 找到最早一次的入院
    minimum_visit_dict = dict()
    for patient_id in visit_info_dict:
        if not minimum_visit_dict.__contains__(patient_id):
            minimum_visit_dict[patient_id] = 100
        for visit_id in visit_info_dict[patient_id]:
            if int(visit_id) < minimum_visit_dict[patient_id]:
                minimum_visit_dict[patient_id] = int(visit_id)
        minimum_visit_dict[patient_id] = str(minimum_visit_dict[patient_id])

    for patient_id in visit_info_dict:
        for visit_id in visit_info_dict[patient_id]:
            minimum_visit_date = visit_info_dict[patient_id][minimum_visit_dict[patient_id]]['入院时间']
            current_date = visit_info_dict[patient_id][visit_id]['入院时间']
            visit_info_dict[patient_id][visit_id]['时间差'] = (current_date-minimum_visit_date).days

    # 最终删除入院时间
    for patient_id in visit_info_dict:
        for visit_id in visit_info_dict[patient_id]:
            visit_info_dict[patient_id][visit_id].pop('入院时间')

    data_to_write = list()
    for patient_id in visit_info_dict:
        for visit_id in visit_info_dict[patient_id]:
            for item in visit_info_dict[patient_id][visit_id]:
                value = visit_info_dict[patient_id][visit_id][item]
                data_to_write.append([patient_id, visit_id, item, value])
    cache_path = os.path.join(cache_root, 'visit_info.csv')
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    return visit_info_dict

def get_procedure(visit_dict, data_path_dict, map_path, cache_root, read_from_cache=False):
    """
    返回病人每一次入院的手术信息
    :param visit_dict:
    :param data_path_dict:
    :param map_path:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        operation_dict = dict()
        cache_path = os.path.join(cache_root, 'operation.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, item, value = line
                if not operation_dict.__contains__(patient_id):
                    operation_dict[patient_id] = dict()
                if not operation_dict[patient_id].__contains__(visit_id):
                    operation_dict[patient_id][visit_id] = dict()
                operation_dict[patient_id][visit_id][item] = value
        return operation_dict

    # 从外源性文件导入名称映射策略
    operation_name_set = set()
    operation_name_list = list()
    with open(map_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            operation_name_set.add(line[0])
            row = []
            for i in range(0, len(line)):
                if len(line[i]) >= 1:
                    row.append(line[i])
            operation_name_list.append(row)

    # 建立数据模板
    operation_dict = dict()
    for patient_id in visit_dict:
        if not operation_dict.__contains__(patient_id):
            operation_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            if not operation_dict[patient_id].__contains__(visit_id):
                operation_dict[patient_id][visit_id] = dict()
            for operation in operation_name_set:
                operation_dict[patient_id][visit_id][operation] = 0

    operation_path = data_path_dict['operation']
    with open(operation_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, _, operation_desc, _, _, _, _, _, _, _, _, _ = line
            if not (operation_dict.__contains__(patient_id) and operation_dict[patient_id].__contains__(visit_id)):
                continue
            for item_list in operation_name_list:
                for i in range(0, len(item_list)):
                    if operation_desc.__contains__(item_list[i]):
                        operation_dict[patient_id][visit_id][item_list[0]] = 1

    data_to_write = list()
    for patient_id in operation_dict:
        for visit_id in operation_dict[patient_id]:
            for item in operation_dict[patient_id][visit_id]:
                value = operation_dict[patient_id][visit_id][item]
                data_to_write.append([patient_id, visit_id, item, value])
    cache_path = os.path.join(cache_root, 'operation.csv')
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    return operation_dict

def get_medical_info(visit_dict, data_path_dict, map_path, cache_root, read_from_cache=False):
    """
    1.记录病人出院后吃了什么药物。判定标准为，某个病人出院前最后36小时中所服用过的所有药物
    2.分类标准，将药物按照外源性文件分为若干大类，只记录每一种大类是否吃药
    :param visit_dict:
    :param data_path_dict:
    :param map_path:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        drug_dict = dict()
        cache_path = os.path.join(cache_root, 'order.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, item, value = line
                if not drug_dict.__contains__(patient_id):
                    drug_dict[patient_id] = dict()
                if not drug_dict[patient_id].__contains__(visit_id):
                    drug_dict[patient_id][visit_id] = dict()
                drug_dict[patient_id][visit_id][item] = value
        return drug_dict

    # 建立药物名称映射
    drug_map_dict = dict()
    drug_map_set = set()
    with open(map_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            drug_map_set.add(line[0])
            for i in range(len(line)):
                if len(line[i]) >= 2:
                    if not drug_map_dict.__contains__(line[i]):
                        drug_map_dict[line[i]] = set()
                    drug_map_dict[line[i]].add(line[0])

    # 对每个病人的每次入院建立相关映射，文档存取了每个病人最后一次用药时间，若没有用药，则时间置为1970年
    drug_dict = dict()
    for patient_id in visit_dict:
        drug_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            drug_dict[patient_id][visit_id] = dict()
            for item in drug_map_set:
                drug_dict[patient_id][visit_id][item] = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)

    # 扫描数据集，获取病人每次出院的出院时间
    discharge_time_dict = dict()
    visit_path = data_path_dict['pat_visit']
    with open(visit_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            if len(line[5]) > 12:
                discharge_time = datetime.datetime.strptime(line[5], '%Y/%m/%d %H:%M:%S')
            elif 4 < len(line[5]) < 12:
                discharge_time = datetime.datetime.strptime(line[5], '%Y/%m/%d')
            else:
                continue

            if not discharge_time_dict.__contains__(patient_id):
                discharge_time_dict[patient_id] = dict()
            discharge_time_dict[patient_id][visit_id] = discharge_time

    # 扫描数据集，获取病人最后一次用药的时间
    order_path = data_path_dict['orders']
    with open(order_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            order_class = line[5]
            order_text = line[6]
            last_acting = line[27]

            if last_acting is None or len(last_acting) < 6:
                continue
            if not (drug_dict.__contains__(patient_id) and drug_dict[patient_id].__contains__(visit_id)):
                continue

            # A代表药疗
            if order_class == 'A':
                for key in drug_map_dict:
                    if order_text.__contains__(key):
                        normalized_name_set = drug_map_dict[key]
                        for normalized_name in normalized_name_set:
                            previous_time = drug_dict[patient_id][visit_id][normalized_name]
                            if len(last_acting) > 12:
                                current_time = datetime.datetime.strptime(last_acting, '%Y/%m/%d %H:%M:%S')
                            elif 4 < len(last_acting) < 12:
                                current_time = datetime.datetime.strptime(last_acting, '%Y/%m/%d')
                            else:
                                continue
                            if current_time > previous_time:
                                drug_dict[patient_id][visit_id][normalized_name] = current_time

    # 如果最后一次实施时间在出院前36小时内，则把相应的时间标注为1
    for patient_id in drug_dict:
        for visit_id in drug_dict[patient_id]:
            discharge_time = discharge_time_dict[patient_id][visit_id]
            for item in drug_dict[patient_id][visit_id]:
                last_perform_time = drug_dict[patient_id][visit_id][item]
                time_interval = discharge_time-last_perform_time
                time_interval = (time_interval.days * 3600 * 24 + time_interval.seconds) / 3600
                if time_interval < 36:
                    drug_dict[patient_id][visit_id][item] = 1
                else:
                    drug_dict[patient_id][visit_id][item] = 0

    cache_path = os.path.join(cache_root, 'order.csv')
    data_to_write = list()
    for patient_id in drug_dict:
        for visit_id in drug_dict[patient_id]:
            for item in drug_dict[patient_id][visit_id]:
                data_to_write.append([patient_id, visit_id, item, drug_dict[patient_id][visit_id][item]])
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data_to_write)

    return drug_dict

def get_vital_sign(visit_dict, data_path_dict, cache_root, read_from_cache=False):
    if read_from_cache:
        vital_sign_dict = dict()
        cache_path = os.path.join(cache_root, 'vital_sign.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, vital_sign, value = line
                if not vital_sign_dict.__contains__(patient_id):
                    vital_sign_dict[patient_id] = dict()
                if not vital_sign_dict[patient_id].__contains__(visit_id):
                    vital_sign_dict[patient_id][visit_id] = dict()
                vital_sign_dict[patient_id][visit_id][vital_sign] = value
        return vital_sign_dict

    # 构建数据模板
    vital_sign_dict = dict()
    feature_list = ['血压Low', '血压high', '身高', '体重', '脉搏', 'BMI']
    for patient_id in visit_dict:
        vital_sign_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            vital_sign_dict[patient_id][visit_id] = dict()
            for item in feature_list:
                vital_sign_dict[patient_id][visit_id][item] = [-1, datetime.datetime(2100, 1, 1, 0, 0, 0, 0)]

    # 读取数据
    vital_sign_path = data_path_dict['vital_signs_rec']
    with open(vital_sign_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, _, time_point, vital_sign, value, _, _, _, _, _, _, _, _, _ = line
            if len(time_point) > 12:
                time_point = datetime.datetime.strptime(time_point, '%Y/%m/%d %H:%M:%S')
            elif 4 < len(time_point) < 12:
                time_point = datetime.datetime.strptime(time_point, '%Y/%m/%d')
            else:
                continue
            if not (vital_sign_dict.__contains__(patient_id) and vital_sign_dict[patient_id].__contains__(visit_id)):
                continue
            if not feature_list.__contains__(vital_sign):
                continue

            _, previous_time = vital_sign_dict[patient_id][visit_id][vital_sign]
            if time_point < previous_time:
                vital_sign_dict[patient_id][visit_id][vital_sign] = [value, time_point]

    # 填写BMI
    for patient_id in vital_sign_dict:
        for visit_id in vital_sign_dict[patient_id]:
            height = float(vital_sign_dict[patient_id][visit_id]['身高'][0])
            weight = float(vital_sign_dict[patient_id][visit_id]['体重'][0])
            if height != 1.0 and weight != -1.0 and height != 1 and weight != -1:
                bmi = weight * 10000 / (height * height)
                if bmi <= 50:
                    vital_sign_dict[patient_id][visit_id]['BMI'][0] = bmi
            vital_sign_dict[patient_id][visit_id].pop('身高')
            vital_sign_dict[patient_id][visit_id].pop('体重')

    # 去除时间信息
    for patient_id in vital_sign_dict:
        for visit_id in vital_sign_dict[patient_id]:
            for vital_sign in vital_sign_dict[patient_id][visit_id]:
                value, _ = vital_sign_dict[patient_id][visit_id][vital_sign]
                vital_sign_dict[patient_id][visit_id][vital_sign] = value

    data_to_write = list()
    for patient_id in vital_sign_dict:
        for visit_id in vital_sign_dict[patient_id]:
            for vital_sign in vital_sign_dict[patient_id][visit_id]:
                value = vital_sign_dict[patient_id][visit_id][vital_sign]
                data_to_write.append([patient_id, visit_id, vital_sign, value])

    cache_path = os.path.join(cache_root, 'vital_sign.csv')
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    return vital_sign_dict

def get_demographic_info(visit_dict, data_path_dict, cache_root, read_from_cache=False):
    """
    :param visit_dict:
    :param data_path_dict:
    :param cache_root:
    :param read_from_cache:
    :return: dict<patient, sex>, dict<patient, dict<visit_id, age>>
    """

    if read_from_cache:
        cache_age_file = os.path.join(cache_root, 'age.csv')
        cache_sex_file = os.path.join(cache_root, 'sex.csv')
        sex_dict_return = dict()
        age_dict_return = dict()

        with open(cache_age_file, 'r', encoding='gbk', newline="") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, age = line
                if not age_dict_return.__contains__(patient_id):
                    age_dict_return[patient_id] = dict()
                age_dict_return[patient_id][visit_id] = age
        with open(cache_sex_file, 'r', encoding='gbk', newline="") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, sex = line
                sex_dict_return[patient_id] = sex
        return sex_dict_return, age_dict_return

    pat_master_path = data_path_dict['pat_master_index']
    # 记录性别与生卒年份
    sex_dict = dict()
    birth_year_dict = dict()
    with open(pat_master_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            sex = line[4]
            birth_year = line[5]

            if sex == '男':
                sex = 1
            elif sex == '女':
                sex = 0
            else:
                continue
            if birth_year is not None and birth_year != "":
                birth_year = int(birth_year[0:4])
            else:
                continue
            sex_dict[patient_id] = sex
            birth_year_dict[patient_id] = birth_year

    # 记录入院时间
    visit_path = data_path_dict['pat_visit']
    visit_year_dict = dict()
    with open(visit_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            admission_year = int(line[3][0:4])

            if not visit_year_dict.__contains__(patient_id):
                visit_year_dict[patient_id] = dict()
            visit_year_dict[patient_id][visit_id] = admission_year

    # 构建返回的两个字典
    sex_dict_return = dict()
    age_dict_return = dict()
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            sex_dict_return[patient_id] = sex_dict[patient_id]
            if not age_dict_return.__contains__(patient_id):
                age_dict_return[patient_id] = dict()
            age_dict_return[patient_id][visit_id] = visit_year_dict[patient_id][visit_id] - birth_year_dict[patient_id]

    # 写缓存
    age_to_write = list()
    for patient_id in age_dict_return:
        for visit_id in age_dict_return[patient_id]:
            age_to_write.append([patient_id, visit_id, age_dict_return[patient_id][visit_id]])
    sex_to_write = list()
    for patient_id in sex_dict_return:
        sex_to_write.append([patient_id, sex_dict_return[patient_id]])

    cache_age_file = os.path.join(cache_root, 'age.csv')
    cache_sex_file = os.path.join(cache_root, 'sex.csv')
    with open(cache_age_file, 'w', encoding="gbk", newline="") as file:
        csv.writer(file).writerows(age_to_write)
    with open(cache_sex_file, 'w', encoding="gbk", newline="") as file:
        csv.writer(file).writerows(sex_to_write)

    return sex_dict_return, age_dict_return

def get_diagnosis_info(visit_dict, data_path_dict, map_path, cache_root, read_from_cache=False):
    """
    返回病人每次入院的主诊断和其它诊断
    :param visit_dict:
    :param data_path_dict:
    :param map_path:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        diagnosis_dict = dict()
        cache_path = os.path.join(cache_root, 'diagnosis.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, diagnosis, value = line
                if not diagnosis_dict.__contains__(patient_id):
                    diagnosis_dict[patient_id] = dict()
                if not diagnosis_dict[patient_id].__contains__(visit_id):
                    diagnosis_dict[patient_id][visit_id] = dict()
                diagnosis_dict[patient_id][visit_id][diagnosis] = value
        return diagnosis_dict

    # 通过外源性归一化清单，建立模板
    normalized_dict = dict()
    normalized_set = set()
    with open(map_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            normalized_set.add(line[0])
            for item in line:
                if len(item) >= 1:
                    normalized_dict[item] = line[0]
    diagnosis_dict = dict()
    for patient_id in visit_dict:
        diagnosis_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            diagnosis_dict[patient_id][visit_id] = dict()
            for item in normalized_set:
                diagnosis_dict[patient_id][visit_id][item] = 0

    # 填充数据
    diagnosis_path = data_path_dict['diagnosis']
    with open(diagnosis_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            diagnosis_type = line[2]
            diagnosis_desc = line[4]

            # 只考虑出院诊断（Type 为3（出院主诊断），A（出院其它诊断）
            if not (diagnosis_type == '3' or diagnosis_type == 'A'):
                continue
            # 丢弃不匹配的数据
            if not (diagnosis_dict.__contains__(patient_id) and diagnosis_dict[patient_id].__contains__(visit_id)):
                continue

            # 进行数据填充
            for item in normalized_dict:
                if diagnosis_desc.__contains__(item):
                    diagnosis_dict[patient_id][visit_id][normalized_dict[item]] = 1

    data_to_write = list()
    for patient_id in diagnosis_dict:
        for visit_id in diagnosis_dict[patient_id]:
            for item in diagnosis_dict[patient_id][visit_id]:
                value = diagnosis_dict[patient_id][visit_id][item]
                data_to_write.append([patient_id, visit_id, item, value])
    cache_path = os.path.join(cache_root, 'diagnosis.csv')
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    return diagnosis_dict

def get_cardiac_dysfunction_info(visit_dict, data_path_dict, cache_root, read_from_cache=False):
    """
    返回病人每次入院的心功能不全分级
    :param visit_dict:
    :param data_path_dict:
    :param map_path:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        diagnosis_dict = dict()
        cache_path = os.path.join(cache_root, 'cardiac_dysfunction.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, diagnosis, value = line
                if not diagnosis_dict.__contains__(patient_id):
                    diagnosis_dict[patient_id] = dict()
                if not diagnosis_dict[patient_id].__contains__(visit_id):
                    diagnosis_dict[patient_id][visit_id] = dict()
                diagnosis_dict[patient_id][visit_id][diagnosis] = value
        return diagnosis_dict

    # 通过外源性归一化清单，建立模板
    diagnosis_dict = dict()
    for patient_id in visit_dict:
        diagnosis_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            diagnosis_dict[patient_id][visit_id] = {'NYHA 1级': 0, 'NYHA 2级': 0, 'NYHA 3级': 0, 'NYHA 4级': 0}

    # 填充数据
    diagnosis_path = data_path_dict['diagnosis']
    with open(diagnosis_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            diagnosis_type = line[2]
            diagnosis_desc = line[4]

            # 只考虑出院诊断（Type 为3（出院主诊断），A（出院其它诊断）
            if not (diagnosis_type == '3' or diagnosis_type == 'A'):
                continue
            # 丢弃不匹配的数据
            if not (diagnosis_dict.__contains__(patient_id) and diagnosis_dict[patient_id].__contains__(visit_id)):
                continue

            # 进行数据填充
            if diagnosis_desc.__contains__('心功能'):
                pos = diagnosis_desc.find('心功能')
                target_length = len('心功能')
                if len(diagnosis_desc) - pos > 8:
                    sub_string = diagnosis_desc[pos + target_length: pos + target_length + 8]
                else:
                    sub_string = diagnosis_desc[pos + target_length:]
                if sub_string.__contains__('1') or sub_string.__contains__('I') or sub_string.__contains__('i') or \
                        sub_string.__contains__('一') or sub_string.__contains__('Ⅰ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 1级'] = 1
                elif sub_string.__contains__('2') or sub_string.__contains__('II') or sub_string.__contains__('ii') or \
                        sub_string.__contains__('二') or sub_string.__contains__('Ⅱ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 2级'] = 1
                elif sub_string.__contains__('3') or sub_string.__contains__('III') or \
                        sub_string.__contains__('iii') or sub_string.__contains__('三') or sub_string.__contains__('Ⅲ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 3级'] = 1
                elif sub_string.__contains__('4') or sub_string.__contains__('IV') or sub_string.__contains__('iv') or \
                        sub_string.__contains__('四') or sub_string.__contains__('Ⅳ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 4级'] = 1
                else:
                    # 单纯写心功能不全的，统一按照二级处理
                    diagnosis_dict[patient_id][visit_id]['NYHA 2级'] = 1
            elif diagnosis_desc.__contains__('NYHA'):
                pos = diagnosis_desc.find('NYHA')
                target_length = len('NYHA')
                if len(diagnosis_desc) - pos > 8:
                    sub_string = diagnosis_desc[pos + target_length: pos + target_length + 8]
                else:
                    sub_string = diagnosis_desc[pos + target_length:]
                if sub_string.__contains__('1') or sub_string.__contains__('I') or sub_string.__contains__('i') or \
                        sub_string.__contains__('一') or sub_string.__contains__('Ⅰ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 1级'] = 1
                elif sub_string.__contains__('2') or sub_string.__contains__('II') or sub_string.__contains__('ii') or \
                        sub_string.__contains__('二') or sub_string.__contains__('Ⅱ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 2级'] = 1
                elif sub_string.__contains__('3') or sub_string.__contains__('III') or \
                        sub_string.__contains__('iii') or sub_string.__contains__('三') or sub_string.__contains__('Ⅲ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 3级'] = 1
                elif sub_string.__contains__('4') or sub_string.__contains__('IV') or sub_string.__contains__('iv') or \
                        sub_string.__contains__('四') or sub_string.__contains__('Ⅳ'):
                    diagnosis_dict[patient_id][visit_id]['NYHA 4级'] = 1
                else:
                    # 单纯写心功能不全的，统一按照二级处理
                    diagnosis_dict[patient_id][visit_id]['NYHA 2级'] = 1

    data_to_write = list()
    for patient_id in diagnosis_dict:
        for visit_id in diagnosis_dict[patient_id]:
            for item in diagnosis_dict[patient_id][visit_id]:
                value = diagnosis_dict[patient_id][visit_id][item]
                data_to_write.append([patient_id, visit_id, item, value])
    cache_path = os.path.join(cache_root, 'cardiac_dysfunction.csv')
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    return diagnosis_dict

def get_lab_test_info(visit_dict, data_path_dict, name_list_path, cache_root, read_from_cache=False):
    """
    返回所有目标Lab_test的入院值，-1代表相关数据缺失
    :param visit_dict:
    :param data_path_dict:
    :param name_list_path:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        lab_test_dict = dict()
        cache_path = os.path.join(cache_root, 'lab_test.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, item, report_item_name, result, unit, abnormal = line
                if not lab_test_dict.__contains__(patient_id):
                    lab_test_dict[patient_id] = dict()
                if not lab_test_dict[patient_id].__contains__(visit_id):
                    lab_test_dict[patient_id][visit_id] = dict()
                data_tuple = [report_item_name, float(result), unit, abnormal]
                lab_test_dict[patient_id][visit_id][item] = data_tuple
        return lab_test_dict

    # 根据外源性数据建立模板
    lab_test_dict = dict()
    name_dict = dict()
    with open(name_list_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            name_dict[line[0]] = line[1]
    for patient_id in visit_dict:
        lab_test_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            lab_test_dict[patient_id][visit_id] = dict()
            for item in name_dict:
                lab_test_dict[patient_id][visit_id][name_dict[item]] = [-1, -1, -1, -1, datetime.datetime(2100, 1, 1, 0, 0, 0, 0)]

    # 整合数据，分别读取Master和Result中的数据，然后分别进行整合
    # 读取Master数据
    master_dict = dict()
    master_path = data_path_dict['lab_test_master']
    valid_test_no = set()
    with open(master_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            test_no = line[0]
            patient_id = line[2]
            visit_id = line[3]
            if not (visit_dict.__contains__(patient_id) and visit_dict[patient_id].__contains__(visit_id)):
                continue

            if not master_dict.__contains__(patient_id):
                master_dict[patient_id] = dict()
            if not master_dict[patient_id].__contains__(visit_id):
                master_dict[patient_id][visit_id] = set()
            master_dict[patient_id][visit_id].add(test_no)
            valid_test_no.add(test_no)

    # 读取Result数据
    result_dict = dict()
    result_path = data_path_dict['lab_result']
    with open(result_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            test_no = line[0]
            item_no = line[1]
            report_item_name = line[3]
            result = line[5]
            unit = line[6]
            abnormal = line[7]
            record_time = line[9]
            if not valid_test_no.__contains__(test_no):
                continue
            if record_time is None or len(record_time) < 6:
                continue
            if len(record_time) > 12:
                record_time = datetime.datetime.strptime(record_time, '%Y/%m/%d %H:%M:%S')
            else:
                record_time = datetime.datetime.strptime(record_time, '%Y/%m/%d')

            if not name_dict.__contains__(report_item_name):
                continue
            if not result_dict.__contains__(test_no):
                result_dict[test_no] = dict()

            if abnormal == 'L':
                abnormal = 0
            elif abnormal == 'N':
                abnormal = 1
            elif abnormal == 'H':
                abnormal = 2
            else:
                abnormal = -1

            result_list = re.findall(r'[-+]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', result)
            if len(result_list) > 0:
                result = result_list[0]
            if len(result_list) == 0 or len(result) == 0:
                result = -1
            result = float(result)
            result_dict[test_no][item_no] = [report_item_name, result, unit, abnormal, record_time]

    # 数据整合
    for patient_id in master_dict:
        for visit_id in master_dict[patient_id]:
            # 若visit_dict无法匹配，则跳过
            if not (visit_dict.__contains__(patient_id) and visit_dict[patient_id].__contains__(visit_id)):
                continue

            for test_no in master_dict[patient_id][visit_id]:
                if not result_dict.__contains__(test_no):
                    continue
                for item_no in result_dict[test_no]:
                    data_tuple = result_dict[test_no][item_no]
                    report_item_name, result, unit, abnormal, record_time = data_tuple
                    if not name_dict.__contains__(report_item_name):
                        continue
                    previous_time = lab_test_dict[patient_id][visit_id][name_dict[report_item_name]][4]
                    if record_time < previous_time:
                        lab_test_dict[patient_id][visit_id][name_dict[report_item_name]] = data_tuple

    cache_path = os.path.join(cache_root, 'lab_test.csv')
    data_to_write = list()
    for patient_id in lab_test_dict:
        for visit_id in lab_test_dict[patient_id]:
            for item in lab_test_dict[patient_id][visit_id]:
                report_item_name, result, unit, abnormal, _ = lab_test_dict[patient_id][visit_id][item]
                data_tuple = [patient_id, visit_id, item, report_item_name, result, unit, abnormal]
                data_to_write.append(data_tuple)
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    return lab_test_dict

def get_egfr(visit_dict, sex_dict, age_dict, lab_test_dict):
    egfr_dict = dict()
    for patient_id in visit_dict:
        egfr_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            age = float(age_dict[patient_id][visit_id])
            sex = float(sex_dict[patient_id])
            scr = lab_test_dict[patient_id][visit_id]['肌酐'][1]
            if age == -1.0 or sex == -1.0 or scr == -1.0:
                egfr_dict[patient_id][visit_id] = -1
                continue

            if sex == 1.0:
                egfr = 186 * ((scr / 88.41) ** -1.154) * (age ** -0.203) * 1
            else:
                egfr = 186 * ((scr / 88.41) ** -1.154) * (age ** -0.203) * 0.742

            egfr_dict[patient_id][visit_id] = egfr
    return egfr_dict

def get_echocardiogram(visit_dict, data_path_dict, map_path, cache_root, read_from_cache=False):
    """
    返回每次病人每一次入院的超声心动图相关数据
    :param visit_dict:
    :param data_path_dict:
    :param map_path:
    :param read_from_cache:
    :param cache_root:
    :return:
    """
    if read_from_cache:
        cache_path = os.path.join(cache_root, 'echocardiogram.csv')
        exam_dict = dict()
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, item, value = line
                if not exam_dict.__contains__(patient_id):
                    exam_dict[patient_id] = dict()
                if not exam_dict[patient_id].__contains__(visit_id):
                    exam_dict[patient_id][visit_id] = dict()
                exam_dict[patient_id][visit_id][item] = value
        return exam_dict

    # 从映射源读取数据
    map_dict = dict()
    feature_set = set()
    with open(map_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            feature_set.add(line[0])
            for item in line:
                if len(item) >= 2:
                    map_dict[item] = line[0]

    # 建立数据模板
    exam_dict = dict()
    for patient_id in visit_dict:
        if not exam_dict.__contains__(patient_id):
            exam_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            if not exam_dict[patient_id].__contains__(visit_id):
                exam_dict[patient_id][visit_id] = dict()
            for item in feature_set:
                exam_dict[patient_id][visit_id][item] = [-1, datetime.datetime(2100, 1, 1, 0, 0, 0, 0)]

    # 首先获取Master信息
    master_path = data_path_dict['exam_master']
    master_dict = dict()
    with open(master_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            exam_no = line[0]
            patient_id = line[3]
            exam_class = line[7]
            exam_sub_class = line[8]
            req_time = line[21]
            visit_id = line[35]
            if len(exam_no) < 1 or len(patient_id) < 1 or len(visit_id) < 1:
                continue
            if not (exam_class == '超声' and exam_sub_class == '心脏'):
                continue
            if len(req_time) > 12:
                req_time = datetime.datetime.strptime(req_time, '%Y/%m/%d %H:%M:%S')
            elif 4 < len(req_time) < 12:
                req_time = datetime.datetime.strptime(req_time, '%Y/%m/%d')
            else:
                continue

            if not master_dict.__contains__(patient_id):
                master_dict[patient_id] = dict()
            if not master_dict[patient_id].__contains__(visit_id):
                master_dict[patient_id][visit_id] = dict()
            master_dict[patient_id][visit_id][exam_no] = req_time

    # 获取result的信息
    result_dict = dict()
    result_path = data_path_dict['exam_report']
    with open(result_path, 'r', encoding='gb18030', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            exam_no = line[0]
            exam_para = line[1]
            if len(exam_para) < 20:
                continue
            result_dict[exam_no] = exam_para

    # 数据整合
    data_dict = dict()
    for patient_id in master_dict:
        if not data_dict.__contains__(patient_id):
            data_dict[patient_id] = dict()
        for visit_id in master_dict[patient_id]:
            if not data_dict[patient_id].__contains__(visit_id):
                data_dict[patient_id][visit_id] = dict()
            for exam_no in master_dict[patient_id][visit_id]:
                time_point = master_dict[patient_id][visit_id][exam_no]
                if result_dict.__contains__(exam_no):
                    data_dict[patient_id][visit_id][exam_no] = [result_dict[exam_no], time_point]
                else:
                    data_dict[patient_id][visit_id][exam_no] = [str(-1), time_point]

    # 数据填充
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            if not (exam_dict.__contains__(patient_id) and exam_dict[patient_id].__contains__(visit_id)):
                continue
            for exam_no in data_dict[patient_id][visit_id]:
                exam_para, time_point = data_dict[patient_id][visit_id][exam_no]
                for item in map_dict:
                    normalized_name = map_dict[item]
                    _, previous_time = exam_dict[patient_id][visit_id][normalized_name]
                    if exam_para.__contains__(item):
                        pos = exam_para.find(item)
                        target_length = len(item)
                        sub_string = exam_para[pos + target_length: pos + target_length + 5]
                        value_list = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', sub_string)
                        if len(value_list) > 0 and time_point < previous_time:
                            exam_dict[patient_id][visit_id][normalized_name] = [value_list[0], time_point]

    data_to_write = list()
    for patient_id in exam_dict:
        for visit_id in exam_dict[patient_id]:
            for item in exam_dict[patient_id][visit_id]:
                value = exam_dict[patient_id][visit_id][item][0]
                exam_dict[patient_id][visit_id][item] = value
                data_to_write.append([patient_id, visit_id, item, value])
    cache_path = os.path.join(cache_root, 'echocardiogram.csv')
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    return exam_dict

def reconstruct_data(visit_dict, visit_info_dict, procedure_dict, exam_dict, sex_dict, age_dict, medical_dict,
                    diagnosis_dict, vital_sign_dict, lab_test_dict, egfr_dict, special_intervention_dict, cardiac_dysfunction_dict,
                    diuretic_history_dict, labtest_binary=False):
    # 获取手术，检查，用药，诊断，关键指标，实验室检查的有序序列
    general_list = list()
    visit_info_list = list()
    procedure_list = list()
    exam_list = list()
    medical_list = list()
    diagnosis_list = list()
    vital_sign_list = list()
    lab_test_list = list()
    special_intervention_list = list()
    cardiac_dysfunction_list = list()
    diuretic_history_list = list()
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            for key in visit_info_dict[patient_id][visit_id]:
                visit_info_list.append(key)
                general_list.append(key)
            general_list.append('年龄')
            general_list.append('性别')
            general_list.append('EGFR')
            for key in procedure_dict[patient_id][visit_id]:
                procedure_list.append(key)
                general_list.append(key)
            for key in exam_dict[patient_id][visit_id]:
                exam_list.append(key)
                general_list.append(key)
            for key in medical_dict[patient_id][visit_id]:
                medical_list.append(key)
                general_list.append(key)
            for key in diagnosis_dict[patient_id][visit_id]:
                diagnosis_list.append(key)
                general_list.append(key)
            for key in vital_sign_dict[patient_id][visit_id]:
                vital_sign_list.append(key)
                general_list.append(key)
            for key in lab_test_dict[patient_id][visit_id]:
                lab_test_list.append(key)
                general_list.append(key)
            for key in special_intervention_dict[patient_id][visit_id]:
                special_intervention_list.append(key)
                general_list.append(key)
            for key in cardiac_dysfunction_dict[patient_id][visit_id]:
                cardiac_dysfunction_list.append(key)
                general_list.append(key)
            for key in diuretic_history_dict[patient_id][visit_id]:
                diuretic_history_list.append(key)
                general_list.append(key)
            break
        break

    # 填充数据
    data_dict = dict()
    for patient_id in visit_dict:
        if not data_dict.__contains__(patient_id):
            data_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            if not data_dict[patient_id].__contains__(visit_id):
                data_dict[patient_id][visit_id] = dict()
            target_dict = data_dict[patient_id][visit_id]
            # 填入其它
            for item in visit_info_list:
                target_dict[item] = visit_info_dict[patient_id][visit_id][item]
            target_dict['年龄'] = age_dict[patient_id][visit_id]
            target_dict['性别'] = sex_dict[patient_id]
            target_dict['EGFR'] = egfr_dict[patient_id][visit_id]
            for item in diagnosis_list:
                target_dict[item] = diagnosis_dict[patient_id][visit_id][item]
            for item in procedure_list:
                target_dict[item] = procedure_dict[patient_id][visit_id][item]
            for item in exam_list:
                target_dict[item] = exam_dict[patient_id][visit_id][item]
            for item in lab_test_list:
                # 此处3为是否异常，1为具体的值
                if labtest_binary:
                    target_dict[item] = lab_test_dict[patient_id][visit_id][item][3]
                else:
                    target_dict[item] = lab_test_dict[patient_id][visit_id][item][1]
            for item in vital_sign_list:
                target_dict[item] = vital_sign_dict[patient_id][visit_id][item]
            for item in medical_list:
                target_dict[item] = medical_dict[patient_id][visit_id][item]
            for item in special_intervention_list:
                target_dict[item] = special_intervention_dict[patient_id][visit_id][item]
            for item in cardiac_dysfunction_list:
                target_dict[item] = cardiac_dysfunction_dict[patient_id][visit_id][item]
            for item in diuretic_history_list:
                target_dict[item] = diuretic_history_dict[patient_id][visit_id][item]

    # 写数据
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in general_list:
        head.append(item)
    data_to_write.append(head)
    for patient_id in data_dict:
        # 强制有序输出
        for i in range(100):
            if data_dict[patient_id].__contains__(str(i)):
                visit_id = str(i)
                row = [patient_id, visit_id]
                for key in general_list:
                    value = data_dict[patient_id][visit_id][key]
                    row.append(value)
                data_to_write.append(row)
    # 挂-1为空
    for i in range(len(data_to_write)):
        for j in range(len(data_to_write[i])):
            if data_to_write[i][j] == -1 or data_to_write[i][j] == '-1':
                data_to_write[i][j] = ''
    save_path = os.path.abspath('resource/未预处理长期纵向数据_离散化_{}.csv'.format(labtest_binary))
    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

def get_special_intervention_info(visit_dict, data_path_dict, map_path, cache_root, read_from_cache=False):
    """
    此处的特殊干预指两项，是否存在血滤事件或者注射利尿剂
    :param visit_dict:
    :param data_path_dict:
    :param cache_root:
    :param read_from_cache:
    :return:
    """
    if read_from_cache:
        drug_dict = dict()
        cache_path = os.path.join(cache_root, 'special_intervention.csv')
        with open(cache_path, 'r', encoding='gbk', newline='') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                patient_id, visit_id, item, value = line
                if not drug_dict.__contains__(patient_id):
                    drug_dict[patient_id] = dict()
                if not drug_dict[patient_id].__contains__(visit_id):
                    drug_dict[patient_id][visit_id] = dict()
                drug_dict[patient_id][visit_id][item] = value
        return drug_dict

    drug_map_dict = dict()
    drug_map_set = set()
    with open(map_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            drug_map_set.add(line[0])
            for i in range(len(line)):
                if len(line[i]) >= 2:
                    if not drug_map_dict.__contains__(line[i]):
                        drug_map_dict[line[i]] = set()
                    drug_map_dict[line[i]].add(line[0])

    drug_dict = dict()
    for patient_id in visit_dict:
        drug_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            drug_dict[patient_id][visit_id] = {'超滤': 0, '注射利尿剂': 0}

    order_path = data_path_dict['orders']
    with open(order_path, 'r', encoding="gbk", newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            visit_id = line[1]
            order_class = line[5]
            order_text = line[6]

            if not (drug_dict.__contains__(patient_id) and drug_dict[patient_id].__contains__(visit_id)):
                continue

            # A代表药疗
            if order_class == 'A':
                for key in drug_map_dict:
                    if order_text.__contains__(key):
                        normalized_name_set = drug_map_dict[key]
                        for normalized_name in normalized_name_set:
                            if normalized_name == '利尿剂' and order_text.__contains__('注射'):
                                drug_dict[patient_id][visit_id]['注射利尿剂'] = 1
            if order_text.__contains__('血液滤过') or order_text.__contains__('血滤') or \
                order_text.__contains__('血液过滤') or order_text.__contains__('超滤'):
                drug_dict[patient_id][visit_id]['超滤'] = 1

    cache_path = os.path.join(cache_root, 'special_intervention.csv')
    data_to_write = list()
    for patient_id in drug_dict:
        for visit_id in drug_dict[patient_id]:
            for item in drug_dict[patient_id][visit_id]:
                data_to_write.append([patient_id, visit_id, item, drug_dict[patient_id][visit_id][item]])
    with open(cache_path, 'w', encoding='gbk', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data_to_write)

    return drug_dict

def diuretic_history(visit_dict, medical_dict):
    diuretic_history_dict = dict()
    for patient_id in visit_dict:
        diuretic_history_dict[patient_id] = dict()
        for visit_id in visit_dict[patient_id]:
            diuretic_history_dict[patient_id][visit_id] = {'利尿剂史': 0}
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            nearest_visit = -1
            for i in range(int(visit_id)):
                if visit_dict[patient_id].__contains__(str(i)):
                    nearest_visit = str(i)
            if nearest_visit == -1:
                continue
            else:
                if medical_dict[patient_id][nearest_visit]['利尿剂'] == '1':
                    diuretic_history_dict[patient_id][visit_id]['利尿剂史'] = '1'
    return diuretic_history_dict
    
def main():
    data_root = os.path.abspath('H:/301HF/Update')
    drug_map_path = os.path.abspath('resource/药品名称映射.csv')
    diagnosis_map_path = os.path.abspath('resource/合并症不同名归一化.csv')
    lab_test_list_path = os.path.abspath('resource/实验室检查名称清单.csv')
    ehcocardiogram_map_path = os.path.abspath('resource/超声心动图名称映射.csv')
    operation_map_path = os.path.abspath('resource/手术名称映射.csv')
    file_name_list = ['diagnosis', 'diagnosis_category', 'exam_items', 'exam_master', 'exam_report', 'lab_result',
                      'lab_test_items', 'lab_test_master', 'operation', 'operation_master', 'operation_name', 'orders',
                      'pat_master_index', 'pat_visit', 'vital_signs_rec']
    data_path_dict = dict()
    for item in file_name_list:
        data_path_dict[item] = os.path.join(data_root, item+'.csv')
    cache_root = os.path.abspath('resource/cache')
    
    visit_dict = get_valid_visit(data_path_dict, cache_root, read_from_cache=True)
    echocardiogram_didct = get_echocardiogram(visit_dict, data_path_dict, ehcocardiogram_map_path, cache_root, read_from_cache=True)
    lab_test_dict = get_lab_test_info(visit_dict, data_path_dict, lab_test_list_path, cache_root, read_from_cache=True)
    diagnosis_dict = get_diagnosis_info(visit_dict, data_path_dict, diagnosis_map_path, cache_root, read_from_cache=True)
    sex_dict, age_dict = get_demographic_info(visit_dict, data_path_dict, cache_root, read_from_cache=True)
    visit_info_dict = get_visit_info(visit_dict, data_path_dict, cache_root, read_from_cache=True)
    operation_dict = get_procedure(visit_dict, data_path_dict, operation_map_path, cache_root, read_from_cache=True)
    vital_sign_dict = get_vital_sign(visit_dict, data_path_dict, cache_root, read_from_cache=True)
    medical_dict = get_medical_info(visit_dict, data_path_dict, drug_map_path, cache_root, read_from_cache=True)
    egfr_dict = get_egfr(visit_dict, sex_dict, age_dict, lab_test_dict)
    cardiac_dysfunction_dict = get_cardiac_dysfunction_info(visit_dict, data_path_dict, cache_root, read_from_cache=True)
    special_intervention_dict = get_special_intervention_info(visit_dict, data_path_dict, drug_map_path, cache_root, read_from_cache=True)
    diuretic_history_dict = diuretic_history(visit_dict, medical_dict)
    reconstruct_data(visit_dict, visit_info_dict, operation_dict, echocardiogram_didct, sex_dict, age_dict, medical_dict,
                    diagnosis_dict, vital_sign_dict, lab_test_dict, egfr_dict, special_intervention_dict, cardiac_dysfunction_dict,
                    diuretic_history_dict, labtest_binary=True)
    print('accomplished')


if __name__ == '__main__':
    main()