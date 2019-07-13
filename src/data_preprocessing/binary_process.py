import csv
import os


def main():
    data_path = os.path.abspath('resource/未预处理长期纵向数据_离散化_True.csv')
    binary_path = os.path.abspath('resource/二值化策略.csv')
    save_data_path = os.path.abspath('resource/二值化后的长期纵向数据.csv')
    data_dict, contextual_list = read_data(data_path)
    binary_process_dict = read_binary_process_feature(binary_path)
    binary_data_dict, revised_content_list = binary(data_dict, binary_process_dict, contextual_list)

    # 写数据
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in revised_content_list:
        head.append(item)
    data_to_write.append(head)

    general_list = head[2:]

    for patient_id in binary_data_dict:
        # 强制有序输出
        for i in range(100):
            if binary_data_dict[patient_id].__contains__(str(i)):
                visit_id = str(i)
                row = [patient_id, visit_id]
                for key in general_list:
                    value = binary_data_dict[patient_id][visit_id][key]
                    row.append(value)
                data_to_write.append(row)
    with open(save_data_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def read_binary_process_feature(data_path):
    """
    目前定义二值化操作的feature为5类
    1. Type 1:原始数据是离散量，根据原始数据可分为“偏高，偏低、正常”，三档的，分为两个哑变量，分别表示偏高的不正常和偏低的不正常
    2. Type 2:原始数据是离散量，根据原始数据可分为“正常，不正常”或“进行、未进行”两档的，创建一个哑变量，表示该值不正常或进行了干预
    3. Type 3:原始数据是连续量，根据原始数据可分为“偏高，偏低、正常”，三档的，分为两个哑变量，分别表示偏高的不正常和偏低的不正常，截断阈值在文件中写明
    4. Type 4:原始数据是连续量，根据原始数据可分为“正常，不正常”两档的，创建一个哑变量，表示该值不正常，截断阈值在文件中写明
    5. Type 5:原始数据是连续量，当数据小于阈值1时，视为非常不正常，小于阈值2时，视为不正常
    6. Type 6:原始数据是连续量，当数据大于阈值2时，视为非常不正常，大于阈值1时，视为不正常
    7. Type 7:不对原始数据做出任何处理（目前仅对时间差一个特征开放）
    
    
    :param data_path:
    :return:
    """
    binary_dict = dict()
    with open(data_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            if line[1] == '1':
                binary_dict[line[0]] = [1]
            elif line[1] == "2":
                binary_dict[line[0]] = [2]
            elif line[1] == '3':
                binary_dict[line[0]] = [3, float(line[2]), float(line[3])]
            elif line[1] == '4':
                binary_dict[line[0]] = [4, line[2], float(line[3])]
            elif line[1] == '5':
                binary_dict[line[0]] = [5, float(line[2]), float(line[3])]
            elif line[1] == '6':
                binary_dict[line[0]] = [6, float(line[2]), float(line[3])]
            elif line[1] == '7':
                binary_dict[line[0]] = [7]
            else:
                raise ValueError('')
    return binary_dict


def binary(data_dict, binary_process_dict, contextual_list):
    """
    根据配置文件进行二值化
    :return:
    """
    item_list = list()

    # 读取item，变换数据结构
    for item in binary_process_dict:
        item_list.append(item)
        contextual_list.remove(item)
        if binary_process_dict[item][0] == 1:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_high"] = 0
                    data_dict[patient_id][visit_id][item + "_low"] = 0

                    value = data_dict[patient_id][visit_id].pop(item)
                    if value == '':
                        pass
                    elif int(value) == 0:
                        data_dict[patient_id][visit_id][item + "_low"] = 1
                    elif int(value) == 1:
                        pass
                    elif int(value) == 2:
                        data_dict[patient_id][visit_id][item + "_high"] = 1
                    else:
                        raise ValueError('')
            contextual_list.append(item + "_low")
            contextual_list.append(item + "_high")
        elif binary_process_dict[item][0] == 2:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_done"] = 0

                    value = data_dict[patient_id][visit_id].pop(item)
                    if value == '':
                        pass
                    elif int(value) == 0:
                        pass
                    elif int(value) == 1:
                        data_dict[patient_id][visit_id][item + "_done"] = 1
                    else:
                        raise ValueError('')
            contextual_list.append(item + "_done")
        elif binary_process_dict[item][0] == 3:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_high"] = 0
                    data_dict[patient_id][visit_id][item + "_low"] = 0

                    value = data_dict[patient_id][visit_id].pop(item)
                    if value == '':
                        pass
                    elif float(value) < float(binary_process_dict[item][1]):
                        data_dict[patient_id][visit_id][item + "_low"] = 1
                    elif float(value) > float(binary_process_dict[item][1]):
                        data_dict[patient_id][visit_id][item + "_high"] = 1
                    else:
                        pass
            contextual_list.append(item + "_low")
            contextual_list.append(item + "_high")
        elif binary_process_dict[item][0] == 4:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_done"] = 0

                    value = data_dict[patient_id][visit_id].pop(item)
                    if value == '':
                        pass
                    elif binary_process_dict[item][1] == -1:
                        if float(value) > float(binary_process_dict[item][2]):
                            data_dict[patient_id][visit_id][item + "_done"] = 1
                    elif binary_process_dict[item][2] == -1:
                        if float(value) < float(binary_process_dict[item][1]):
                            data_dict[patient_id][visit_id][item + "_done"] = 1
                    else:
                        raise ValueError('')
            contextual_list.append(item + "_done")
        elif binary_process_dict[item][0] == 5:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_abnormal"] = 0
                    data_dict[patient_id][visit_id][item + "_very_abnormal"] = 0

                    value = data_dict[patient_id][visit_id].pop(item)
                    if value == '':
                        pass
                    elif float(value) < float(binary_process_dict[item][1]):
                        data_dict[patient_id][visit_id][item + "_very_abnormal"] = 1
                    elif float(value) < float(binary_process_dict[item][2]):
                        data_dict[patient_id][visit_id][item + "_abnormal"] = 1
                    else:
                        pass
            contextual_list.append(item + "_abnormal")
            contextual_list.append(item + "_very_abnormal")
        elif binary_process_dict[item][0] == 6:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_abnormal"] = 0
                    data_dict[patient_id][visit_id][item + "_very_abnormal"] = 0

                    value = data_dict[patient_id][visit_id].pop(item)
                    if value == '':
                        pass
                    elif float(value) > float(binary_process_dict[item][2]):
                        data_dict[patient_id][visit_id][item + "_very_abnormal"] = 1
                    elif float(value) > float(binary_process_dict[item][1]):
                        data_dict[patient_id][visit_id][item + "_abnormal"] = 1
                    else:
                        pass
            contextual_list.append(item + "_abnormal")
            contextual_list.append(item + "_very_abnormal")
        elif binary_process_dict[item][0] == 7:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item] = data_dict[patient_id][visit_id].pop(item)
            contextual_list.append(item)
        else:
            raise ValueError('')
    return data_dict, contextual_list


def read_data(file_path):
    data_dict = dict()
    feature_dict = dict()
    context_index_list = list()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    feature_dict[i] = line[i]
                    # 按照需求记录event, context的有序内容信息
                    context_index_list.append(line[i])
                head_flag = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            data_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                data_dict[patient_id][visit_id][feature_dict[i]] = line[i]
    return data_dict, context_index_list


if __name__ == '__main__':
    main()
