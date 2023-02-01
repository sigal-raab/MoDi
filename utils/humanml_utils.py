import numpy as np

op2hm_dict = {0: 15, 1: 12, 2: 16, 3: 18, 4: 20, 5: 17, 6: 19, 7: 21, 8: 0, 9: 1, 10: 4, 11: 7, 12: 2, 13: 5, 14: 8,
              15: 9, 16: 6, 17: 13, 18: 14, 19: 3, 20: 22}  # note: 7 & 8 are new in (4,10) & (8,11) respectively
hm2op_dict = {15: 0, 12: 1, 16: 2, 18: 3, 20: 4, 17: 5, 19: 6, 21: 7, 0: 8, 1: 9, 4: 10, 7: 11, 2: 12, 5: 13, 8: 14,
              9: 15, 6: 16, 13: 17, 14: 18, 3: 19, 22: 20, 10: 11, 11: 14}

human_ml_len = 22
openpose_len = 20

switch_hands = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                13: 14, 16: 17, 18: 19, 20: 21, 14: 13, 17: 16, 19: 18, 21: 20, 15: 15,  22: 22}

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


dicop = {(8, 19): (8, 20), (16, 19): (8, 19), (15, 16): (16, 19), (1, 15): (15, 16),
                           (0, 1): (1, 15), (15, 18): (15, 16), (5, 18): (15, 18), (5, 6): (5, 18), (6, 7): (5, 6),
                           (15, 17): (15, 16), (2, 17): (15, 17), (2, 3): (2, 17), (3, 4): (2, 3), (8, 12): (8, 20),
                           (12, 13): (8, 12), (13, 14): (12, 13), (8, 9): (8, 20), (9, 10): (8, 9), (10, 11): (9, 10)}

dichm = {(0, 3): (0, 22), (6, 3): (0, 3), (9, 6): (6, 3), (12, 9): (9, 6),
                           (15, 12): (12, 9), (9, 13): (9, 6), (16, 13): (9, 13), (16, 18): (16, 13),
                           (18, 20): (16, 18), (9, 14): (9, 6), (17, 14): (9, 14), (17, 19): (17, 14),
                           (19, 21): (17, 19), (0, 2): (0, 22), (2, 5): (0, 2), (5, 8): (2, 5), (8, 11): (5, 8),
                           (0, 1): (0, 22), (1, 4): (0, 1), (4, 7): (1, 4), (7, 10): (4, 7)}

print(openpose_tuple_dict_to_humanml(dicop))

op2hm_dict = switch_hands
hm2op_dict = switch_hands

print(openpose_tuple_dict_to_humanml(dichm))
