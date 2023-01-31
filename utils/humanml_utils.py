import numpy as np

op2hm_dict = {0: 15, 1: 12, 2: 17, 3: 19, 4: 21, 5: 16, 6: 18, 7: 20, 8: 0, 9: 1, 10: 4, 11: 7, 12: 2, 13: 5, 14: 8,
              15: 9, 16: 6, 17: 14, 18: 13, 19: 3, 20: 22}  # note: 7 & 8 are new in (4,10) & (8,11) respectively
hm2op_dict = {15: 0, 12: 1, 17: 2, 19: 3, 21: 4, 16: 5, 18: 6, 20: 7, 0: 8, 1: 9, 4: 10, 10: 11, 2: 12, 5: 13, 11: 14,
              9: 15, 6: 16, 14: 17, 13: 18, 3: 19, 22: 20, 7: 11, 8: 14}

human_ml_len = 22
openpose_len = 20


def openpose_list_to_humanml(idxs):
    return [op2hm_dict[idx] for idx in idxs]


def openpose_tuple_to_humanml(idxs):
    return tuple(op2hm_dict[idx] for idx in idxs)


def openpose_list_dict_vals_to_humanml(dic):
    # for skeletal pooling
    new_dic = {}
    for k in dic.keys():
        new_dic[k] = openpose_list_to_humanml(dic[k])
    return new_dic


def reorder_openpose_to_human_ml(array):
    return [(array[hm2op_dict[i]] if hm2op_dict[i] < len(array) else None) for i in range(openpose_len)]


def openpose_tuples_to_humanml(tuples):
    return [openpose_tuple_to_humanml(t) for t in tuples]


def openpose_tuples_dict_vals_to_humanml(dic):
    # for skeletal pooling
    new_dic = {}
    for k in dic.keys():
        new_dic[k] = openpose_tuples_to_humanml(dic[k])
    return new_dic


def openpose_tuple_dict_to_humanml(dic):
    new_dic = {}
    for k in dic.keys():
        new_dic[openpose_tuple_to_humanml(k)] = openpose_tuple_to_humanml(dic[k])
    return new_dic


dic =  { (8, 19): (8, 20), (16, 19): (8, 19), (15, 16): (16, 19), (1, 15): (15, 16),
                           (0, 1): (1, 15), (15, 18): (15, 16), (5, 18): (15, 18), (5, 6): (5, 18), (6, 7): (5, 6),
                           (15, 17): (15, 16), (2, 17): (15, 17), (2, 3): (2, 17), (3, 4): (2, 3), (8, 12): (8, 20),
                           (12, 13): (8, 12), (13, 14): (12, 13), (8, 9): (8, 20), (9, 10): (8, 9), (10, 11): (9, 10)}

print(openpose_tuple_dict_to_humanml(dic))
