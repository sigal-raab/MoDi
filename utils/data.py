import os.path as osp
from glob import glob
import numpy as np
import pandas as pd
import copy
from itertools import chain

from Motion.AnimationStructure import children_list, get_sorted_order
from Motion.Quaternions import Quaternions
from Motion.Animation import Animation

import torch  # used for foot contact
from utils.foot import get_foot_contact
import prepare_modi_data

foot_names_openpose = ['LeftFoot', 'RightFoot']
foot_names = ['LeftAnkle', 'LeftToes', 'RightAnkle', 'RightToes']


class openpose_joints(object):
    def __init__(self):
        super().__init__()

        self.oredered_joint_names = \
            np.array(['chin', 'collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
                      'pelvis', 'r_heap', 'r_knee', 'r_ankle', 'l_heap', 'l_knee', 'l_ankle'])
        self.parent_joint_names = \
            np.array(['collar', 'pelvis', 'collar', 'r_shoulder', 'r_elbow', 'collar', 'l_shoulder', 'l_elbow',
                      np.nan, 'pelvis', 'r_heap', 'r_knee', 'pelvis', 'l_heap', 'l_knee'])

    def name2idx(self, joint_names):
        multiple_names = isinstance(joint_names, np.ndarray)
        if not multiple_names:
            joint_names = np.array([joint_names])
        indices = np.zeros(joint_names.shape, dtype=np.int)
        for i, name in enumerate(joint_names):
            try:
                idx = np.where(self.oredered_joint_names == name)[0][0]
            except:
                idx = -1
            indices[i] = idx
        if not multiple_names:
            indices = indices[0]
        return indices

    def idx2name(self, joint_idx):
        return self.oredered_joint_names[joint_idx]


class humanml_joints(openpose_joints):
    def __init__(self):
        super().__init__()

        self.oredered_joint_names = \
            np.array(prepare_modi_data.SMPL_JOINT_NAMES)
        self.parent_joint_names = \
            np.array([np.nan, 'Pelvis', 'Pelvis', 'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
                      'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'Spine3', 'Spine3', 'Neck', 'L_Collar', 'R_Collar',
                      'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow'])


def entity2index(obj, ordered_entities):
    """
    convert a data struct (dict, list) of entities to a dict of their indices.
    :param obj: object of entitities
    :param ordered_entities: all possible entities, structured like obj.
    example: obj = {'bird': 44, 'cat': 12}, ordered_entities = {['cat', 'dog', 'bird']: [12, 44, 55]}
            result = {2: 1, 0: 0}
    :return:
    """
    if isinstance(obj, dict):
        assert len(ordered_entities) == 2
        ordered_keys = ordered_entities[0]
        ordered_values = ordered_entities[1]
        res = {ordered_keys.index(key): entity2index(val, ordered_values) for key, val in obj.items()}
    elif isinstance(obj, list):
        ordered_values = ordered_entities[0]
        res = [entity2index(item, ordered_values) for item in obj]
    else:
        res = ordered_entities.index(obj)
    return res


class OpenposeJoint():
    n_channels = 3

    @staticmethod
    def str():
        return 'Joint'

    skeletal_pooling_dist_1 = [{0: [0, 1]},
                               # root: [collar,pelvis]
                               {0: [0, 1, 2, 3], 1: [0, 3, 4, 5]},
                               # collar:[collar,r_wrist,l_wrist,pelvis], pelvis:[pelvis,collar,r_ankle,l_ankle]
                               {0: [0, 1, 3, 5], 1: [1, 2], 2: [3, 4], 3: [0, 5, 6, 8], 4: [6, 7], 5: [8, 9]},
                               {0: [0, 1, 2, 5], 1: [2, 3], 2: [3, 4], 3: [5, 6], 4: [6, 7], 5: [8, 9, 12], 6: [9, 10],
                                7: [10, 11], 8: [12, 13], 9: [13, 14]}]
    skeletal_pooling_dist_0 = [{0: [1]},
                               {0: [0], 1: [3]},
                               {0: [0], 1: [2], 2: [4], 3: [5], 4: [7], 5: [9]},
                               {0: [1], 1: [3], 2: [4], 3: [6], 4: [7], 5: [8], 6: [10], 7: [11], 8: [13], 9: [14]}]
    oj = openpose_joints()
    parents_list = [[-1],
                    [-1, 0],
                    [3, 0, 0, -1, 3, 3],
                    [5, 0, 1, 0, 3, -1, 5, 6, 5, 8],
                    oj.name2idx(oj.parent_joint_names)]  # full skeleton possesses openpose topology
    parents_list = list(map(np.array, parents_list))  # convert list items to np


class Joint(OpenposeJoint):
    skeletal_pooling_dist_1 = [{0: [0, 1]},
                               # root: [collar,pelvis]
                               {0: [0, 1, 2, 3], 1: [0, 3, 4, 5]},
                               # collar:[collar,r_wrist,l_wrist,pelvis], pelvis:[pelvis,collar,r_ankle,l_ankle]
                               {0: [0, 1, 3, 5], 1: [1, 2], 2: [3, 4], 3: [0, 5, 6, 8], 4: [6, 7], 5: [8, 9]},
                               {0: [15, 12, 16, 17], 1: [16, 18], 2: [18, 20], 3: [17, 19], 4: [19, 21], 5: [0, 1, 2],
                                6: [1, 4], 7: [4, 7, 10], 8: [2, 5], 9: [5, 8, 11]}]
    skeletal_pooling_dist_0 = [{0: [1]},
                               {0: [0], 1: [3]},
                               {0: [0], 1: [2], 2: [4], 3: [5], 4: [7], 5: [9]},
                               {0: [12], 1: [18], 2: [20], 3: [19], 4: [21], 5: [0], 6: [4], 7: [10], 8: [5], 9: [11]}]
    hj = humanml_joints()
    parents_list = [[-1],
                    [-1, 0],
                    [3, 0, 0, -1, 3, 3],
                    [5, 0, 1, 0, 3, -1, 5, 6, 5, 8],
                    hj.name2idx(hj.parent_joint_names)]  # full skeleton possesses openpose topology

    parents_list = list(map(np.array, parents_list))  # convert list items to np


class OpenPoseEdge():
    n_channels = 4

    @staticmethod
    def str():
        return 'Edge'

    edges_list = [
        [(0, 1)],
        [(1, 2), (0, 1)],
        [(3, 7), (0, 1), (0, 2), (0, 3), (0, 6), (3, 4), (3, 5)],
        [(5, 12), (0, 1), (1, 2), (0, 3), (3, 4), (0, 11), (5, 11), (5, 6), (6, 7), (5, 8), (8, 9), (0, 10)],
        # order of ground truth:
        [(8, 20), (8, 19), (16, 19), (15, 16), (1, 15), (0, 1), (15, 18), (5, 18), (5, 6), (6, 7), (15, 17), (2, 17),
         (2, 3), (3, 4), (8, 12), (12, 13), (13, 14), (8, 9), (9, 10), (10, 11)]
    ]
    skeletal_pooling_dist_1_edges = [
        {(0, 1): [(0, 1), (1, 2)]},
        {(1, 2): [(3, 7)], (0, 1): [(0, 1), (0, 2), (0, 3), (0, 6), (3, 4), (3, 5)]},
        {(3, 7): [(5, 12)], (0, 1): [(0, 1), (1, 2)], (0, 2): [(0, 3), (3, 4)], (0, 3): [(0, 11), (5, 11)],
         (0, 6): [(0, 10)],
         (3, 4): [(5, 6), (6, 7)], (3, 5): [(5, 8), (8, 9)]},
        {(5, 12): [(8, 20)], (0, 1): [(15, 17), (2, 17)], (1, 2): [(2, 3), (3, 4)], (0, 3): [(15, 18), (5, 18)],
         (3, 4): [(5, 6), (6, 7)],
         (0, 11): [(15, 16), (1, 15)], (5, 11): [(8, 19), (16, 19)], (5, 6): [(8, 9)], (6, 7): [(9, 10), (10, 11)],
         (5, 8): [(8, 12)], (8, 9): [(12, 13), (13, 14)], (0, 10): [(0, 1)]}]
    assert all([list(parents.keys()) == edges for (parents, edges) in
                zip(skeletal_pooling_dist_1_edges, edges_list)])  # same order as edges_list
    assert all([set(edges) == set(chain.from_iterable(pooling_list.values())) for edges, pooling_list in
                zip(edges_list[1:], skeletal_pooling_dist_1_edges)])  # all target edges are covered while up sampling
    assert all([set(edges) == set(pooling_list.keys()) for edges, pooling_list in
                zip(edges_list[:-1], skeletal_pooling_dist_1_edges)])  # all source edges are covered while up sampling

    skeletal_pooling_dist_0_edges = [
        {(0, 1): [(1, 2)]},
        {(1, 2): [(3, 7)], (0, 1): [(0, 3)]},
        {(3, 7): [(5, 12)], (0, 1): [(1, 2)], (0, 2): [(3, 4)], (0, 3): [(5, 11)], (0, 6): [(0, 10)],
         (3, 4): [(6, 7)], (3, 5): [(8, 9)]},
        {(5, 12): [(8, 20)], (0, 1): [(2, 17)], (1, 2): [(3, 4)], (0, 3): [(5, 18)], (3, 4): [(6, 7)],
         (0, 11): [(15, 16)], (5, 11): [(8, 19)], (5, 6): [(8, 9)], (6, 7): [(10, 11)],
         (5, 8): [(8, 12)], (8, 9): [(13, 14)], (0, 10): [(0, 1)]}]
    assert all([list(parents.keys()) == edges for (parents, edges) in
                zip(skeletal_pooling_dist_0_edges, edges_list)])  # same order as edges_list

    parents_list_edges = [{(0, 1): -1},
                          {(1, 2): -1, (0, 1): (1, 2)},
                          {(3, 7): -1, (0, 1): (0, 3), (0, 2): (0, 3), (0, 3): (3, 7), (0, 6): (0, 3), (3, 4): (3, 7),
                           (3, 5): (3, 7)},
                          {(5, 12): -1, (0, 1): (0, 11), (1, 2): (0, 1), (0, 3): (0, 11), (3, 4): (0, 3),
                           (0, 11): (5, 11), (5, 11): (5, 12),
                           (5, 6): (5, 12), (6, 7): (5, 6), (5, 8): (5, 12), (8, 9): (5, 8), (0, 10): (0, 11)},
                          {(8, 20): -1, (8, 19): (8, 20), (16, 19): (8, 19), (15, 16): (16, 19), (1, 15): (15, 16),
                           (0, 1): (1, 15), (15, 18): (15, 16), (5, 18): (15, 18), (5, 6): (5, 18), (6, 7): (5, 6),
                           (15, 17): (15, 16), (2, 17): (15, 17), (2, 3): (2, 17), (3, 4): (2, 3), (8, 12): (8, 20),
                           (12, 13): (8, 12), (13, 14): (12, 13), (8, 9): (8, 20), (9, 10): (8, 9), (10, 11): (9, 10)}]
    assert all([list(parents.keys()) == edges for (parents, edges) in
                zip(parents_list_edges, edges_list)])  # same order as edges_list
    assert all([list(parents_list.values())[0] == -1 for parents_list in parents_list_edges])  # 1st item is root

    feet_list_edges = [[], [(0, 1)], [(3, 5), (3, 4)], [(8, 9), (6, 7)],
                       [(13, 14), (10, 11)]]  # should be ordered [LeftFoot,RightFoot]

    n_hierarchical_stages = len(parents_list_edges)
    assert len(edges_list) == n_hierarchical_stages and len(parents_list_edges) == n_hierarchical_stages and \
           len(feet_list_edges) == n_hierarchical_stages

    # edge values to edge indices
    skeletal_pooling_dist_1 = [entity2index(pooling_edges, [smaller_edges, [larger_edges]])
                               for pooling_edges, smaller_edges, larger_edges
                               in zip(skeletal_pooling_dist_1_edges, edges_list[:-1], edges_list[1:])]
    skeletal_pooling_dist_0 = [entity2index(pooling_edges, [smaller_edges, [larger_edges]])
                               for pooling_edges, smaller_edges, larger_edges
                               in zip(skeletal_pooling_dist_0_edges, edges_list[:-1], edges_list[1:])]
    parents_list = [entity2index(item, [edges, edges]) for item, edges in
                    zip(parents_list_edges, [edges + [-1] for edges in edges_list])]
    # restore the -1
    parents_list = [[val if val != len(edges) else -1 for val in parents.values()] for parents, edges in
                    zip(parents_list, edges_list)]
    feet_idx_list = [entity2index(item, [edges, edges]) for item, edges in
                     zip(feet_list_edges, [edges + [-1] for edges in edges_list])]
    n_edges = [len(parents) for parents in parents_list]

    @classmethod
    def is_global_position_enabled(cls):
        return all(-2 in parents for parents in cls.parents_list)

    @classmethod
    def enable_global_position(cls):
        """ add a special entity that would be the global position.
        The entity is appended to the edges list.
        No need to really add it in edges_list and all the other structures that are based on tupples. We add it only
        to the structures that are based on indices.
        Its neighboring edges are the same as the neightbors of root """

        if cls.is_global_position_enabled():
            # if enable_global_position has already been called before, do nothing
            return

        for pooling_list in [cls.skeletal_pooling_dist_0, cls.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage, n_edges_this_hierarchical_stage, n_edges_larger_hierarchical_stage, parents in \
                    zip(pooling_list, cls.n_edges[:-1], cls.n_edges[1:], cls.parents_list):
                # last entry in current hierarchy pools from last entry in larger hierarchy
                pooling_hierarchical_stage[n_edges_this_hierarchical_stage] = [n_edges_larger_hierarchical_stage]
        for parents in cls.parents_list:
            # new entry's 'parent' would be -2
            parents.append(-2)
        cls.n_edges = [len(parents) for parents in cls.parents_list]  # num edges after adding root

    @classmethod
    def enable_repr6d(cls):
        cls.n_channels = 6

    @classmethod
    def enable_marker4(cls):
        cls.n_channels = 12

    @classmethod
    def is_foot_contact_enabled(cls, level=-1):
        # return all(any([isinstance(parent, tuple) and parent[0] == -3 for parent in parents])
        #            for parents in cls.parents_list)
        return any([isinstance(parent, tuple) and parent[0] == -3 for parent in cls.parents_list[level]])

    @classmethod
    def enable_foot_contact(cls):
        """ add special entities that would be the foot contact labels.
        The entities are appended to the edges list.
        No need to really add them in edges_list and all the other structures that are based on tuples. We add them only
        to the structures that are based on indices.
        Their neighboring edges are the same as the neightbors of the feet """

        if cls.is_foot_contact_enabled():
            # if enable_global_position has already been called before, do nothing
            return

        for hierarchical_stage_idx, (feet_idx, parents) in enumerate(zip(cls.feet_idx_list, cls.parents_list)):
            for idx, foot in enumerate(feet_idx):
                # new entry's 'parent' would be a tuple (-3, foot)
                parents.append((-3, foot))

                if hierarchical_stage_idx < cls.n_hierarchical_stages - 1:  # update pooling only for stages lower than last
                    last_idx_this = cls.n_edges[hierarchical_stage_idx] + idx
                    last_idx_larger = cls.n_edges[hierarchical_stage_idx + 1] + idx
                    for pooling_list in [cls.skeletal_pooling_dist_0, cls.skeletal_pooling_dist_1]:
                        # last entry in current hierarchy pools from last entry in larger hierarchy
                        pooling_list[hierarchical_stage_idx][last_idx_this] = [last_idx_larger]
        cls.n_edges = [len(parents) for parents in cls.parents_list]  # num edges after adding feet


class Edge(OpenPoseEdge):
    edges_list = [
        [(0, 1)],
        [(1, 2), (0, 1)],
        [(3, 7), (0, 1), (0, 2), (0, 3), (0, 6), (3, 4), (3, 5)],
        [(5, 12), (0, 1), (1, 2), (0, 3), (3, 4), (0, 11), (5, 11), (5, 6), (6, 7), (5, 8), (8, 9), (0, 10)],
        # order of ground truth:
        [(0, 22), (0, 3), (6, 3), (9, 6), (12, 9), (15, 12), (9, 14), (17, 14), (17, 19), (19, 21), (9, 13), (16, 13),
         (16, 18), (18, 20), (0, 2), (2, 5), (5, 8), (8, 11), (0, 1), (1, 4), (4, 7), (7, 10)]

    ]
    skeletal_pooling_dist_1_edges = [
        {(0, 1): [(0, 1), (1, 2)]},
        {(1, 2): [(3, 7)], (0, 1): [(0, 1), (0, 2), (0, 3), (0, 6), (3, 4), (3, 5)]},
        {(3, 7): [(5, 12)], (0, 1): [(0, 1), (1, 2)], (0, 2): [(0, 3), (3, 4)], (0, 3): [(0, 11), (5, 11)],
         (0, 6): [(0, 10)],
         (3, 4): [(5, 6), (6, 7)], (3, 5): [(5, 8), (8, 9)]},
        {(5, 12): [(0, 22)], (0, 1): [(9, 13), (16, 13)], (1, 2): [(16, 18), (18, 20)], (0, 3): [(9, 14), (17, 14)],
         (3, 4): [(17, 19), (19, 21)], (0, 11): [(9, 6), (12, 9)], (5, 11): [(0, 3), (6, 3)], (5, 6): [(0, 1)],
         (6, 7): [(1, 4), (4, 7), (7, 10)], (5, 8): [(0, 2)], (8, 9): [(2, 5), (5, 8), (8, 11)], (0, 10): [(15, 12)]}]
    assert all([list(parents.keys()) == edges for (parents, edges) in
                zip(skeletal_pooling_dist_1_edges, edges_list)])  # same order as edges_list
    assert all([set(edges) == set(chain.from_iterable(pooling_list.values())) for edges, pooling_list in
                zip(edges_list[1:], skeletal_pooling_dist_1_edges)])  # all target edges are covered while up sampling
    assert all([set(edges) == set(pooling_list.keys()) for edges, pooling_list in
                zip(edges_list[:-1], skeletal_pooling_dist_1_edges)])  # all source edges are covered while up sampling

    skeletal_pooling_dist_0_edges = [
        {(0, 1): [(1, 2)]},
        {(1, 2): [(3, 7)], (0, 1): [(0, 3)]},
        {(3, 7): [(5, 12)], (0, 1): [(1, 2)], (0, 2): [(3, 4)], (0, 3): [(5, 11)], (0, 6): [(0, 10)],
         (3, 4): [(6, 7)], (3, 5): [(8, 9)]},
        {(5, 12): [(0, 22)], (0, 1): [(16, 13)], (1, 2): [(18, 20)], (0, 3): [(17, 14)], (3, 4): [(19, 21)],
         (0, 11): [(9, 6)], (5, 11): [(0, 3)], (5, 6): [(0, 1)], (6, 7): [(4, 7)], (5, 8): [(0, 2)], (8, 9): [(5, 8)],
         (0, 10): [(15, 12)]}]
    assert all([list(parents.keys()) == edges for (parents, edges) in
                zip(skeletal_pooling_dist_0_edges, edges_list)])  # same order as edges_list

    parents_list_edges = [{(0, 1): -1},
                          {(1, 2): -1, (0, 1): (1, 2)},
                          {(3, 7): -1, (0, 1): (0, 3), (0, 2): (0, 3), (0, 3): (3, 7), (0, 6): (0, 3), (3, 4): (3, 7),
                           (3, 5): (3, 7)},
                          {(5, 12): -1, (0, 1): (0, 11), (1, 2): (0, 1), (0, 3): (0, 11), (3, 4): (0, 3),
                           (0, 11): (5, 11), (5, 11): (5, 12),
                           (5, 6): (5, 12), (6, 7): (5, 6), (5, 8): (5, 12), (8, 9): (5, 8), (0, 10): (0, 11)},
                          {(0, 22): -1, (0, 3): (0, 22), (6, 3): (0, 3), (9, 6): (6, 3), (12, 9): (9, 6),
                           (15, 12): (12, 9), (9, 14): (9, 6), (17, 14): (9, 14), (17, 19): (17, 14),
                           (19, 21): (17, 19), (9, 13): (9, 6), (16, 13): (9, 13), (16, 18): (16, 13),
                           (18, 20): (16, 18), (0, 2): (0, 22), (2, 5): (0, 2), (5, 8): (2, 5), (8, 11): (5, 8),
                           (0, 1): (0, 22), (1, 4): (0, 1), (4, 7): (1, 4), (7, 10): (4, 7)}
                          ]
    assert all([list(parents.keys()) == edges for (parents, edges) in
                zip(parents_list_edges, edges_list)])  # same order as edges_list
    assert all([list(parents_list.values())[0] == -1 for parents_list in parents_list_edges])  # 1st item is root

    feet_list_edges = [[], [(0, 1)], [(3, 5), (3, 4)], [(8, 9), (6, 7)],
                       # should be ordered ['LeftAnkle', 'LeftToes', 'RightAnkle', 'RightToes']
                       [(5, 8), (8, 11), (4, 7), (7, 10)]]

    n_hierarchical_stages = len(parents_list_edges)
    assert len(edges_list) == n_hierarchical_stages and len(parents_list_edges) == n_hierarchical_stages and \
           len(feet_list_edges) == n_hierarchical_stages

    # edge values to edge indices
    skeletal_pooling_dist_1 = [entity2index(pooling_edges, [smaller_edges, [larger_edges]])
                               for pooling_edges, smaller_edges, larger_edges
                               in zip(skeletal_pooling_dist_1_edges, edges_list[:-1], edges_list[1:])]
    skeletal_pooling_dist_0 = [entity2index(pooling_edges, [smaller_edges, [larger_edges]])
                               for pooling_edges, smaller_edges, larger_edges
                               in zip(skeletal_pooling_dist_0_edges, edges_list[:-1], edges_list[1:])]
    parents_list = [entity2index(item, [edges, edges]) for item, edges in
                    zip(parents_list_edges, [edges + [-1] for edges in edges_list])]
    # restore the -1
    parents_list = [[val if val != len(edges) else -1 for val in parents.values()] for parents, edges in
                    zip(parents_list, edges_list)]
    feet_idx_list = [entity2index(item, [edges, edges]) for item, edges in
                     zip(feet_list_edges, [edges + [-1] for edges in edges_list])]
    n_edges = [len(parents) for parents in parents_list]


def collect_motions_loc(root_path):
    """ Read mixamo npy files from disk and concatente to one single file. Save together with a file of motion paths (character / motion_type / sub_motion_idx)"""
    npy_files = glob(osp.join(root_path, '*', '*', 'motions', '*.npy'))

    motion_shape = np.load(npy_files[0]).shape  # just to get motion shape
    assert motion_shape[0] == 15
    motion_shape = tuple(
        np.array(motion_shape) + np.array([1, 0, 0]))  # make sure the number of joints is an exponent of 2
    all_motions = np.zeros((len(npy_files),) + motion_shape)
    all_motion_names = list()
    for idx, file in enumerate(npy_files):
        all_motions[idx, 0:15] = np.load(file)
        all_motion_names.append(file)

    np.save(osp.join(root_path, 'motion.npy'), all_motions)
    all_motion_names = np.array(all_motion_names)
    for idx, motion_name in enumerate(all_motion_names):
        all_motion_names[idx] = osp.relpath(motion_name, root_path)
    np.savetxt(osp.join(root_path, 'motion_order.txt'), all_motion_names, fmt='%s')


def calc_bone_lengths(motion_data, parents=None, names=None):
    """

    :param motion_data: shape: [#motions, #joints(15-16), #axes(3), #frames]
    :return:
    """

    if motion_data.ndim == 3:  # no batch dim
        motion_data = motion_data[np.newaxis]

    # support 16 joints skeleton
    if motion_data.shape[1] == 16:
        motion_data = motion_data[:, :15]

    n_joints = motion_data.shape[1]
    if parents is None or names is None:
        opj = openpose_joints()
        is_openpose = n_joints in [len(opj.oredered_joint_names), len(opj.oredered_joint_names) + 1]
        if parents is None:
            if is_openpose:
                parents = opj.name2idx(opj.parent_joint_names)
            else:
                raise 'Cannot determine bone length with no hierarchy info'
        if names is None:
            if is_openpose:
                names = opj.oredered_joint_names
            else:
                names = np.array([*map(str, np.arange(n_joints))])

    bone_lengths = pd.DataFrame(index=names, columns=['mean', 'std'])
    for joint_idx in range(n_joints):
        parent_idx = parents[joint_idx]
        joint_name = names[joint_idx]
        if parent_idx not in [None, -1]:
            all_bone_lengths = np.linalg.norm(motion_data[:, joint_idx, :, :] - motion_data[:, parent_idx, :, :],
                                              axis=1)
            bone_lengths['mean'][joint_name] = all_bone_lengths.mean()
            bone_lengths['std'][joint_name] = all_bone_lengths.std()
        else:
            bone_lengths.drop(joint_name, inplace=True)

    return bone_lengths


def neighbors_by_distance(parents, dist=1):
    assert dist in [0, 1], 'distance larger than 1 is not supported yet'

    neighbors = {joint_idx: [joint_idx] for joint_idx in range(len(parents))}

    if dist == 1:  # code should be general to any distance. for now dist==1 is the largest supported

        # handle non virtual joints
        n_entities = len(parents)
        children = children_list(parents)
        for joint_idx in range(n_entities):
            parent_idx = parents[joint_idx]
            if parent_idx not in [-1, -2] and not isinstance(parent_idx,
                                                             tuple):  # -1 is the parent of root. -2 is the parent of global location, tuple for foot_contact
                neighbors[joint_idx].append(parent_idx)  # add entity's parent
            neighbors[joint_idx].extend(children[joint_idx])  # append all entity's children

        # handle global pos virtual joint
        glob_pos_exists = Edge.is_global_position_enabled()
        if glob_pos_exists:
            root_idx = parents.index(-1)
            glob_pos_idx = parents.index(-2)

            # global position should have same neighbors of root and should become his neighbors' neighbor
            neighbors[glob_pos_idx].extend(neighbors[root_idx])
            for root_neighbor in neighbors[root_idx]:
                # changing the neighbors of root during iteration puts the new neighbor in the iteration
                if root_neighbor != root_idx:
                    neighbors[root_neighbor].append(glob_pos_idx)
            neighbors[root_idx].append(glob_pos_idx)  # finally change root itself

        # handle foot contact label virtual joint
        foot_contact_exists = Edge.is_foot_contact_enabled()
        if foot_contact_exists:
            foot_and_contact_label = [(i, parents[i][1]) for i in range(len(parents)) if
                                      isinstance(parents[i], tuple) and parents[i][0] == -3]

            # 'contact' joint should have same neighbors of related joint and should become his neighbors' neighbor
            for foot_idx, contact_label_idx in foot_and_contact_label:
                neighbors[contact_label_idx].extend(neighbors[foot_idx])
                for foot_neighbor in neighbors[foot_idx]:
                    # changing the neighbors of root during iteration puts the new neighbor in the iteration
                    if foot_neighbor != foot_idx:
                        neighbors[foot_neighbor].append(contact_label_idx)
                neighbors[foot_idx].append(contact_label_idx)  # finally change foot itself

    return neighbors


def expand_topology_edges(anim, req_joint_idx=None, names=None, offset_len_mean=None, nearest_joint_ratio=0.9):
    assert nearest_joint_ratio == 1, 'currently not supporting nearest_joint_ratio != 1'

    # we do not want to change inputs that are given as views
    anim = copy.deepcopy(anim)
    req_joint_idx = copy.deepcopy(req_joint_idx)
    names = copy.deepcopy(names)
    offset_len_mean = copy.deepcopy(offset_len_mean)

    n_frames, n_joints_all = anim.shape
    if req_joint_idx is None:
        req_joint_idx = np.arange(n_joints_all)
    if names is None:
        names = np.array([str(i) for i in range(len(req_joint_idx))])

    # fix the topology according to required joints
    parent_req = np.zeros(n_joints_all)  # fixed parents according to req
    n_children_req = np.zeros(n_joints_all)  # number of children per joint in the fixed topology
    children_all = children_list(anim.parents)
    for idx in req_joint_idx:
        child = idx
        parent = anim.parents[child]
        while parent not in np.append(req_joint_idx, -1):
            child = parent
            parent = anim.parents[child]
        parent_req[idx] = parent
        if parent != -1:
            n_children_req[parent] += 1

    # find out how many joints have multiple children
    super_parents = np.where(n_children_req > 1)[0]
    n_super_children = n_children_req[super_parents].sum().astype(int)  # total num of multiple children

    if n_super_children == 0:
        return anim, req_joint_idx, names, offset_len_mean  # can happen in lower hierarchy levels

    # prepare space for expanded joints, at the end of each array
    anim.offsets = np.append(anim.offsets, np.zeros(shape=(n_super_children, 3)), axis=0)
    anim.positions = np.append(anim.positions, np.zeros(shape=(n_frames, n_super_children, 3)), axis=1)
    anim.rotations = Quaternions(np.append(anim.rotations, Quaternions.id((n_frames, n_super_children)), axis=1))
    anim.orients = Quaternions(np.append(anim.orients, Quaternions.id(n_super_children), axis=0))
    anim.parents = np.append(anim.parents, np.zeros(n_super_children, dtype=int))
    names = np.append(names, np.zeros(n_super_children, dtype='<U40'))
    if offset_len_mean is not None:
        offset_len_mean = np.append(offset_len_mean, np.zeros(n_super_children))

    # fix topology and names
    new_joint_idx = n_joints_all
    req_joint_idx = np.append(req_joint_idx, new_joint_idx + np.arange(n_super_children))
    for parent in super_parents:
        for child in children_all[parent]:
            anim.parents[new_joint_idx] = parent
            anim.parents[child] = new_joint_idx
            names[new_joint_idx] = names[parent] + '_' + names[child]

            new_joint_idx += 1

    # sort data items in a topological order
    sorted_order = get_sorted_order(anim.parents)
    anim = anim[:, sorted_order]
    names = names[sorted_order]
    if offset_len_mean is not None:
        offset_len_mean = offset_len_mean[sorted_order]

    # assign updated locations to req_joint_idx
    sorted_order_inversed = {num: i for i, num in enumerate(sorted_order)}
    sorted_order_inversed[-1] = -1
    req_joint_idx = np.array([sorted_order_inversed[i] for i in req_joint_idx])
    req_joint_idx.sort()

    return anim, req_joint_idx, names, offset_len_mean


def expand_topology_joints(one_motion_data, is_openpose, parents=None, names=None):
    # one_motion_data: frame x joint x axis
    if is_openpose:
        one_motion_data, parents, names = expand_topology_joints_openpose(one_motion_data)
    else:
        one_motion_data, parents, names = expand_topology_joints_general(one_motion_data, parents, names)

    return one_motion_data, parents, names


def expand_topology_joints_general(one_motion_data, parents, names=None, nearest_joint_ratio=0.9):
    n_frames = one_motion_data.shape[0]
    n_joints = one_motion_data.shape[1]
    n_axes = one_motion_data.shape[2]
    assert n_joints == len(parents)

    if names is None:
        names = np.array([str(i) for i in range(n_joints)])
    other_joint_ratio = 1 - nearest_joint_ratio

    children = children_list(parents)
    # super_parents are joints that have more than one child
    super_parents = [parent for parent, ch in enumerate(children) if len(ch) > 1]
    n_multiple_children = sum([children[parent].shape[0] for parent in super_parents])
    one_motion_data = np.concatenate([one_motion_data, np.zeros((n_frames, n_multiple_children, n_axes))], axis=1)
    parents = np.append(parents, np.zeros(n_multiple_children, dtype=int))
    names = np.append(names, np.zeros(n_multiple_children, dtype='<U5'))

    new_joint_idx = n_joints
    for parent in super_parents:
        for child in children[parent]:
            one_motion_data[:, new_joint_idx, :] = nearest_joint_ratio * one_motion_data[:, parent, :] + \
                                                   other_joint_ratio * one_motion_data[:, child, :]
            parents[new_joint_idx] = parent
            parents[child] = new_joint_idx
            names[new_joint_idx] = names[parent] + '_' + str(child)
            new_joint_idx += 1

    return one_motion_data, parents, names


def expand_topology_joints_openpose(one_motion_data, nearest_joint_ratio=0.9):
    other_joint_ratio = 1 - nearest_joint_ratio

    oj = openpose_joints()
    oj.oredered_joint_names = np.concatenate(
        [oj.oredered_joint_names,
         np.array(['r_pelvis', 'l_pelvis', 'u_pelvis', 'r_collar', 'l_collar', 'u_collar'])])
    oj.parent_joint_names = np.concatenate(
        [oj.parent_joint_names, np.array(['pelvis', 'pelvis', 'pelvis', 'collar', 'collar', 'collar'])])
    oj.parent_joint_names[oj.name2idx('chin')] = 'u_collar'
    oj.parent_joint_names[oj.name2idx('r_shoulder')] = 'r_collar'
    oj.parent_joint_names[oj.name2idx('l_shoulder')] = 'l_collar'
    oj.parent_joint_names[oj.name2idx('collar')] = 'u_pelvis'
    oj.parent_joint_names[oj.name2idx('r_heap')] = 'r_pelvis'
    oj.parent_joint_names[oj.name2idx('l_heap')] = 'l_pelvis'

    parents = oj.name2idx(oj.parent_joint_names)

    n_frames = one_motion_data.shape[0]
    n_axes = one_motion_data.shape[2]
    one_motion_data = np.concatenate([one_motion_data[:, :15, :], np.zeros((n_frames, 6, n_axes))], axis=1)
    one_motion_data[:, oj.name2idx('r_pelvis'), :] = \
        nearest_joint_ratio * one_motion_data[:, oj.name2idx('pelvis'), :] + \
        other_joint_ratio * one_motion_data[:, oj.name2idx('r_heap'), :]
    one_motion_data[:, oj.name2idx('l_pelvis'), :] = \
        nearest_joint_ratio * one_motion_data[:, oj.name2idx('pelvis'), :] + \
        other_joint_ratio * one_motion_data[:, oj.name2idx('l_heap'), :]
    one_motion_data[:, oj.name2idx('u_pelvis'), :] = \
        nearest_joint_ratio * one_motion_data[:, oj.name2idx('pelvis'), :] + \
        other_joint_ratio * one_motion_data[:, oj.name2idx('collar'), :]
    one_motion_data[:, oj.name2idx('r_collar'), :] = \
        nearest_joint_ratio * one_motion_data[:, oj.name2idx('collar'), :] + \
        other_joint_ratio * one_motion_data[:, oj.name2idx('r_shoulder'), :]
    one_motion_data[:, oj.name2idx('l_collar'), :] = \
        nearest_joint_ratio * one_motion_data[:, oj.name2idx('collar'), :] + \
        other_joint_ratio * one_motion_data[:, oj.name2idx('l_shoulder'), :]
    one_motion_data[:, oj.name2idx('u_collar'), :] = \
        nearest_joint_ratio * one_motion_data[:, oj.name2idx('collar'), :] + \
        other_joint_ratio * one_motion_data[:, oj.name2idx('chin'), :]
    one_motion_data -= one_motion_data[0, oj.name2idx('pelvis'),
                       :]  # subtract pelvis of the 1st frame so joints will be around (0,0,0)
    names = oj.oredered_joint_names

    return one_motion_data, parents, names


def to_list_4D(motion_data):
    if not isinstance(motion_data, list):
        assert motion_data.ndim in [3, 4]
        if motion_data.ndim == 3:
            motion_data = motion_data[np.newaxis]
        motion_data = list(np.expand_dims(motion_data, 1))
    assert is_list_4D(motion_data)
    return motion_data


def to_cpu(motion_data):
    if motion_data is None:
        return None
    if not isinstance(motion_data, list):
        motion_data = motion_data.cpu()
    else:
        for i, motion in enumerate(motion_data):
            motion_data[i] = to_cpu(motion)
    return motion_data


def un_normalize(data, mean, std):
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = un_normalize(data[i], mean, std)
    else:
        if data.shape[1] == std.shape[1]:
            data = data * std + mean
    return data


def is_list_4D(data):
    return isinstance(data, list) and all([d.ndim == 4 and d.shape[0] == 1 for d in data])


def anim_from_edge_motion_data(edge_motion_data, edge_rot_dict_general):
    edge_rots_dict, _, _ = edge_rot_dict_from_edge_motion_data(edge_motion_data,
                                                               edge_rot_dict_general=edge_rot_dict_general)
    n_motions = edge_motion_data.shape[0]
    anims = [None] * n_motions
    for idx, one_edge_rot_dict in enumerate(edge_rots_dict):
        anims[idx], names = anim_from_edge_rot_dict(one_edge_rot_dict)
    return anims, names


def anim_from_edge_rot_dict(edge_rot_dict, root_name='Hips'):
    assert root_name in edge_rot_dict['names_with_root']
    root_idx = np.where(edge_rot_dict['names_with_root'] == root_name)[0][0]
    n_frames = edge_rot_dict['rot_edge_no_root'].shape[0]
    n_joints = edge_rot_dict['rot_edge_no_root'].shape[1] + 1  # add one for the root
    offsets = np.insert(edge_rot_dict['offsets_no_root'], root_idx, edge_rot_dict['offset_root'], axis=0)
    positions = np.repeat(offsets[np.newaxis], n_frames, axis=0)
    positions[:, root_idx] = edge_rot_dict['pos_root']
    parents = edge_rot_dict['parents_with_root']
    orients = Quaternions.id(n_joints)
    rotations = Quaternions(np.insert(edge_rot_dict['rot_edge_no_root'], root_idx, edge_rot_dict['rot_root'], axis=1))

    if rotations.shape[-1] == 6:  # repr6d
        from Motion.transforms import repr6d2quat
        rotations = repr6d2quat(rotations)
    anim_edges = Animation(rotations, positions, orients, offsets, parents)

    sorted_order = get_sorted_order(anim_edges.parents)
    anim_edges_sorted = anim_edges[:, sorted_order]
    names_sorted = edge_rot_dict['names_with_root'][sorted_order]

    # expand joints
    anim_exp, _, names_exp, _ = expand_topology_edges(anim_edges_sorted, names=names_sorted, nearest_joint_ratio=1)

    # move rotation values to parents
    children_all_joints = children_list(anim_exp.parents)
    for idx, children_one_joint in enumerate(children_all_joints[1:]):
        parent_idx = idx + 1
        if len(children_one_joint) > 0:  # not leaf
            assert len(children_one_joint) == 1 or (anim_exp.offsets[children_one_joint] == np.zeros(3)).all() and (
                    anim_exp.rotations[:, children_one_joint] == Quaternions.id((n_frames, 1))).all()
            anim_exp.rotations[:, parent_idx] = anim_exp.rotations[:, children_one_joint[0]]
        else:
            anim_exp.rotations[:, parent_idx] = Quaternions.id((n_frames))
    return anim_exp, names_exp


def edge_rot_dict_from_edge_motion_data(motion_data, type='sample', edge_rot_dict_general=None):
    if isinstance(motion_data, np.ndarray) and motion_data.ndim == 4 and motion_data.shape[0] > 1:
        # several non-sub-motions
        edge_rots = [None] * motion_data.shape[0]
        frames_mult = [1] * motion_data.shape[0]
        is_sub_motion = False
        for i, motion in enumerate(motion_data):
            edge_rots_internal, _, _ = edge_rot_dict_from_edge_motion_data(motion, type, edge_rot_dict_general)
            assert len(edge_rots_internal) == 1
            edge_rots[i] = edge_rots_internal[0]
        return edge_rots, frames_mult, is_sub_motion

    assert is_list_4D(motion_data)
    is_sub_motion = type in ['sample', 'interp-mix-pyramid'] and (len(motion_data) > 1)
    edge_rots = [None] * len(motion_data)
    frame_mults = [1] * len(motion_data)
    glob_pos = Edge.is_global_position_enabled()
    feet = Edge.is_foot_contact_enabled()

    # store upper level values
    offsets = edge_rot_dict_general['offsets_no_root']
    names = edge_rot_dict_general['names_with_root']
    parents = Edge.parents_list[
        -1]  # upper level parents. to be used in case there is no inner iteration on hierarcy levels
    assert all(edge_rot_dict_general['parents_with_root'] == Edge.parents_list[-1][:len(edge_rot_dict_general[
                                                                                            'parents_with_root'])])  # use 'len' because root pose may be added to Edge.parents_list[-1]
    n_frames_max = motion_data[-1].shape[-1]
    n_feet = len(Edge.feet_list_edges[-1])  # use this in case there is no pyramid

    # handle all levels
    for hierarchy_level in range(len(motion_data) - 1, -1,
                                 -1):  # iterate in reverse order so each level can use its upper one
        motion = motion_data[hierarchy_level]
        motion_tr = motion[0].transpose(2, 0, 1)  # edges x features x frames ==> frames x edges x features
        n_frames = motion_tr.shape[0]
        n_edges = motion_tr.shape[1]
        frame_mults[hierarchy_level] = int(n_frames_max / n_frames)
        if type in ['sample', 'interp-mix-pyramid'] and n_edges != Edge.n_edges[-1]:
            # and hierarchy_level != len(motion_data) - 1:  # uppermost hierarchy level
            n_feet = len(Edge.feet_list_edges[hierarchy_level])  # override the n_feet from outside the loop
            parents = Edge.parents_list[hierarchy_level]
            nearest_edge_idx_w_root_edge = np.array(
                (list(Edge.skeletal_pooling_dist_0[hierarchy_level].values()))).flatten()
            assert parents[0] == -1 and Edge.parents_list[hierarchy_level + 1][
                0] == -1  # make next line count on root location at index 0
            nearest_edge_idx = nearest_edge_idx_w_root_edge[
                               1:] - 1  # root is first, hence we look from index 1 and reduce one because root is first on uppler level too.
            if feet and n_feet > 0:  # n_feet is 0 even when feet are used, for the lowermost level
                # remove foot contact label
                nearest_edge_idx = nearest_edge_idx[:-n_feet]
                nearest_edge_idx_w_root_edge = nearest_edge_idx_w_root_edge[:-n_feet]
            if glob_pos:
                # remove root pos 'edge'
                nearest_edge_idx = nearest_edge_idx[:-1]
                nearest_edge_idx_w_root_edge = nearest_edge_idx_w_root_edge[:-1]
            offsets = offsets[nearest_edge_idx]
            names = names[nearest_edge_idx_w_root_edge]
        if feet and n_feet > 0:  # n_feet is 0 even when feet are used, for the lowermost level
            # last edges are feet contact labels
            rot_edge_no_feet = motion_tr[:, :-n_feet, :]  # drop root rotation (1st idx). it will be used in 'rot_root'
            parents_no_feet = parents[:-n_feet]
            feet_label = motion_tr[:, -n_feet:, :]
        else:
            rot_edge_no_feet = motion_tr
            parents_no_feet = parents
            feet_label = None
        rot_edge_no_root = rot_edge_no_feet[:, 1:, :]  # drop root rotation (1st idx). it will be used in 'rot_root'
        if glob_pos:
            # last edge is global position.
            rot_edge_no_root = rot_edge_no_root[:, :-1,
                               :]  # drop root rotation (1st idx). it will be used in 'rot_root'
            pose_root = rot_edge_no_feet[:, -1, :3]  # drop the 4th item in the position tensor
            parents_no_pose = parents_no_feet[:-1]
            if 'use_velocity' in edge_rot_dict_general and edge_rot_dict_general['use_velocity']:
                pose_root = np.cumsum(pose_root, axis=0)
        else:
            pose_root = np.zeros((n_frames, 3))
            parents_no_pose = parents_no_feet

        edge_rots[hierarchy_level] = {'rot_edge_no_root': rot_edge_no_root,
                                      'parents_with_root': parents_no_pose,
                                      # edge_rot_dict_general['parents_with_root'],
                                      'offsets_no_root': offsets,  # edge_rot_dict_general['offsets_no_root'],
                                      'rot_root': motion_tr[:, 0],
                                      'pos_root': pose_root,
                                      'offset_root': np.zeros(3),
                                      'names_with_root': names,
                                      'contact': feet_label}
        # repeat frames so the frame number of the is_sub_motion would be the same as the final one,
        # in order to make the visualization synchronized
        for element in ['rot_edge_no_root', 'rot_root', 'pos_root']:
            edge_rots[hierarchy_level][element] = \
                edge_rots[hierarchy_level][element].repeat(frame_mults[hierarchy_level], axis=0)

    return edge_rots, frame_mults, is_sub_motion


def motion_from_raw(args, motion_data_raw):
    if args.entity == 'Joint':
        if args.skeleton:
            # motion data has an additional dummy joint to support prev architecture. we don't need it anymore so we drop it
            motion_data = motion_data_raw[:, :15]
        else:
            motion_data = motion_data_raw
        # compute mean, maybe add a motion 'Joint' like in 2D-Motion-Retargeting?
        motion_data -= motion_data[:, 8:9, :, :]  # locate body center at 0,0,0
        if args.normalize:
            mean_joints = motion_data.mean(axis=(0, 3))
            mean_joints = mean_joints[np.newaxis, :, :, np.newaxis]
            std_joints = motion_data.std(axis=(0, 3))
            std_joints = std_joints[np.newaxis, :, :, np.newaxis]
            std_joints[np.where(std_joints < 1e-9)] = 1e-9
            assert (std_joints >= 1e-9).all()
            motion_data = (motion_data - mean_joints) / std_joints
        else:
            mean_joints = np.zeros((1, motion_data.shape[1], motion_data.shape[2], 1))
            std_joints = np.ones_like(mean_joints)
        edge_rot_dict_general = None
    else:  # entity == 'Edge'
        edge_rot_dicts = copy.deepcopy(motion_data_raw)
        edge_rot_dict_general = edge_rot_dicts[0]
        edge_rot_data = np.stack([motion['rot_edge_no_root'] for motion in edge_rot_dicts])
        root_rot_data = np.stack([motion['rot_root'] for motion in edge_rot_dicts])

        if args.rotation_repr == 'repr6d':
            from Motion.transforms import quat2repr6d
            edge_rot_data = quat2repr6d(edge_rot_data)
            root_rot_data = quat2repr6d(root_rot_data)
        motion_data = np.concatenate([root_rot_data[:, :, np.newaxis], edge_rot_data], axis=2)

        if args.glob_pos:  # enable global position
            root_pos_data = np.stack([motion['pos_root'] for motion in edge_rot_dicts])
            root_pos_data -= root_pos_data[:, :1]  # reduce 1st frame pos so motion would start from 0,0,0
            if args.use_velocity:
                root_pos_data[:, 1:] = root_pos_data[:, 1:] - root_pos_data[:, :-1]
            if True:  # to be changed to "if quaternions" once using other rotation representations (6D, euler...)
                # turn 3D pose into 4D by padding with zeros, to match quaternion dims
                dim_delta = motion_data.shape[-1] - root_pos_data.shape[-1]
                root_pos_data = np.concatenate([root_pos_data, np.zeros(root_pos_data.shape[:-1] + (dim_delta,))],
                                               axis=2)
            if args.use_velocity:
                edge_rot_dict_general['use_velocity'] = True
            else:
                edge_rot_dict_general['use_velocity'] = False
            motion_data = np.concatenate([motion_data, root_pos_data[:, :, np.newaxis]], axis=2)

        # samples x frames x edges x features ==> samples x edges x features x frames
        motion_data = motion_data.transpose(0, 2, 3, 1)

        if args.normalize:
            mean_joints = motion_data.mean(axis=(0, 3))
            mean_joints = mean_joints[np.newaxis, :, :, np.newaxis]
            std_joints = motion_data.std(axis=(0, 3))
            std_joints = std_joints[np.newaxis, :, :, np.newaxis]
            std_joints[np.where(std_joints < 1e-9)] = 1e-9
            assert (std_joints >= 1e-9).all()
            motion_data = (motion_data - mean_joints) / std_joints
        else:
            mean_joints = np.zeros((1, motion_data.shape[1], motion_data.shape[2], 1))
            std_joints = np.ones_like(mean_joints)

        # Trick to make denormalize work for foot contact extraction
        edge_rot_dict_general['mean'] = mean_joints.transpose(0, 2, 1, 3)
        edge_rot_dict_general['std'] = std_joints.transpose(0, 2, 1, 3)

        if args.foot:
            motion_data_torch = torch.from_numpy(motion_data).transpose(1, 2)
            motion_data_torch = append_foot_contact(motion_data_torch, args.glob_pos, args.axis_up,
                                                    edge_rot_dict_general)
            motion_data = motion_data_torch.transpose(1, 2).numpy()

            # Do not normalize the contact label
            padding = np.zeros((1, motion_data.shape[1] - mean_joints.shape[1], motion_data.shape[2], 1))
            mean_joints = np.append(mean_joints, padding, axis=1)
            std_joints = np.append(std_joints, np.ones_like(padding), axis=1)

        # Normalization in the shape of (1, n_feature, n_joints, 1)
        edge_rot_dict_general['mean'] = mean_joints.transpose(0, 2, 1, 3)
        edge_rot_dict_general['std'] = std_joints.transpose(0, 2, 1, 3)

        edge_rot_dict_general['mean_tensor'] = torch.tensor(edge_rot_dict_general['mean'], dtype=torch.float32).cuda()
        edge_rot_dict_general['std_tensor'] = torch.tensor(edge_rot_dict_general['std'], dtype=torch.float32).cuda()

    # motion_data = motion_data[:1,...]; assert args.batch == 1; print('*********\nOVERFITTING!!!\n******') # overfit over one sample only

    return motion_data, mean_joints, std_joints, edge_rot_dict_general


def append_foot_contact(motion_data, glob_pos, axis_up, edge_rot_dict_general):
    foot_contact = get_foot_contact(motion_data, glob_pos, axis_up, edge_rot_dict_general, foot_names=foot_names)

    #  pad foot_contact to the size of motion_data features (quaternions or other)
    n_foot = len(foot_names)
    foot_contact_padded = torch.zeros_like(motion_data[:, :, :n_foot])
    foot_contact_padded[:, 0] = foot_contact

    # concatenate on edges axis (foot_contact is like an extra edge)
    motion_data = torch.cat((motion_data, foot_contact_padded), dim=2)

    return motion_data


if __name__ == "__main__":
    pass
