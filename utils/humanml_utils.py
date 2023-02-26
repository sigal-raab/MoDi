import os.path

import numpy as np
import torch
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_quaternion
from Motion import Animation
from Motion import AnimationStructure
from Motion import InverseKinematics
from Motion.Quaternions import Quaternions
from Motion import BVH
from abc import abstractmethod
import random

DATASET_BASE_PATH = r"D:\Documents\University\DeepGraphicsWorkshop\git\HumanML3D\HumanML3D"
HUMANML_JOINT_NAMES = [
    'Pelvis',  # 0
    'L_Hip',  # 1
    'R_Hip',  # 2
    'Spine1',  # 3
    'L_Knee',  # 4
    'R_Knee',  # 5
    'Spine2',  # 6
    'L_Ankle',  # 7
    'R_Ankle',  # 8
    'Spine3',  # 9
    'L_Foot',  # 10
    'R_Foot',  # 11
    'Neck',  # 12
    'L_Collar',  # 13
    'R_Collar',  # 14
    'Head',  # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',  # 18
    'R_Elbow',  # 19
    'L_Wrist',  # 20
    'R_Wrist',  # 21
]
HUMANML_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
IDENTITY_ROTATION = [1., 0., 0., 0., 1., 0.]


class HumanMLFrame:
    def __init__(self, sample_vec, joints_num=22):
        self.data_vec = sample_vec
        self.joints_num = joints_num


class HumanMLVecFrame(HumanMLFrame):

    @property
    def root_rot_velocity(self):
        return self.data_vec[0]

    @property
    def root_linear_velocity(self):
        return self.data_vec[1:3]

    @property
    def root_y(self):
        return self.data_vec[3]

    @property
    def ric_data(self):
        return np.concatenate([np.array([0, self.root_y, 0]).reshape((1, 3)),
                               np.reshape(self.data_vec[4:67], (self.joints_num - 1, 3))], axis=0)

    @property
    def rot_data(self):
        return np.concatenate([np.array(IDENTITY_ROTATION).reshape((1, 6)),
                               np.reshape(self.data_vec[67:193], (self.joints_num - 1, 6))], axis=0)

    def get_rot_data_quat(self):
        """returns the expected torch tensor"""
        return Quaternions(matrix_to_quaternion(rotation_6d_to_matrix(torch.from_numpy(self.rot_data))).numpy())

    @property
    def foot_contact(self):
        return self.data_vec[259:263]

    def get_foot_contact(self):
        """returns the expected torch tensor"""
        return torch.from_numpy(self.foot_contact)


class HumanMLPosFrame(HumanMLFrame):
    @property
    def positions(self):
        return self.data_vec.reshape((self.joints_num, 3))


class HumanMLSample:
    def __init__(self, path, sample_num=None, is_mirror=False, joints_num=22):
        """if sample_num is given, path will be regarded as base path"""
        if sample_num is None:
            self.path = path
        else:
            sample_id = "%06d" % sample_num
            if is_mirror:
                sample_id = 'M' + sample_id
            self.path = os.path.join(path, sample_id + '.npy')
        self.joints_num = joints_num
        self.samples_vec = np.load(self.path)
        self._frames_num = self.samples_vec.shape[0]

    @abstractmethod
    def get_frame(self, frame_num):
        pass

    def __getitem__(self, key):
        return self.get_frame(key)

    def __iter__(self):
        self.cur_frame_num = 0
        return self

    def __next__(self):
        if self.cur_frame_num < self.length:
            frame = self.get_frame(self.cur_frame_num)
            self.cur_frame_num += 1
            return frame
        else:
            raise StopIteration

    @property
    def length(self):
        return self._frames_num


class HumanMLVecSample(HumanMLSample):
    def get_frame(self, frame_num):
        return HumanMLVecFrame(self.samples_vec[frame_num])

    @property
    def root_rot_velocity(self):
        return self.samples_vec[:, 0]

    @property
    def root_linear_velocity(self):
        return self.samples_vec[:, 1:3]

    @property
    def root_y(self):
        return self.samples_vec[:, 3]

    @property
    def ric_data(self):
        return np.concatenate([np.stack([np.zeros_like(self.root_y), self.root_y, np.zeros_like(self.root_y)],
                                        axis=1).reshape((self.length, 1, 3)),
                               np.reshape(self.samples_vec[:, 4:67], (self.length, self.joints_num - 1, 3))], axis=1)

    @property
    def rot_data(self):
        return np.concatenate([np.array(IDENTITY_ROTATION * self.length).reshape((self.length, 1, 6)),
                               np.reshape(self.samples_vec[:, 67:193], (self.length, self.joints_num - 1, 6))], axis=1)

    def get_rot_data_quat(self):
        """returns the expected torch tensor"""
        return Quaternions(matrix_to_quaternion(rotation_6d_to_matrix(torch.from_numpy(self.rot_data))).numpy())

    @property
    def foot_contact(self):
        return self.samples_vec[:, 259:263]

    def get_foot_contact(self):
        """returns the expected torch tensor"""
        return torch.from_numpy(self.foot_contact)


class HumanMLPosSample(HumanMLSample):
    def __init__(self, path, sample_num=None, is_mirror=False, joints_num=22):
        super().__init__(path, sample_num, is_mirror, joints_num)
        self.samples_vec = self.samples_vec.reshape([-1, self.joints_num, 3])
        self._frames_num = self.samples_vec.shape[0]

    def get_frame(self, frame_num):
        return HumanMLPosFrame(self.samples_vec[frame_num])

    @property
    def positions(self):
        return self.samples_vec.reshape((self.length, self.joints_num, 3))


class SampleReader:

    def __init__(self):
        reference_rot_sample, reference_pos_sample = self.get_vec_pos_samples(DATASET_BASE_PATH, 0)
        self.reference_rot = reference_rot_sample[0].get_rot_data_quat()
        self.reference_ric = reference_rot_sample[0].ric_data
        self.reference_pos = reference_pos_sample[0].positions
        self.reference_offsets = self.pos_to_offsets(self.reference_pos)
        self.reference_orients = self.rot_to_orient(self.reference_rot)
        self.sorted_order = AnimationStructure.get_sorted_order(HUMANML_PARENTS)

    @classmethod
    def pos_to_offsets(cls, pos):
        loc_for_offsets = pos
        offsets = loc_for_offsets - loc_for_offsets[HUMANML_PARENTS]
        offsets[0] = loc_for_offsets[0]
        return offsets

    @classmethod
    def rot_to_orient(cls, rot):
        rel_rot = rot / rot[HUMANML_PARENTS]
        rel_rot[0] = rot[0]
        return rel_rot

    @staticmethod
    def get_vec_pos_samples(base_path, sample_num, is_mirror=False):
        vec_path = os.path.join(base_path, "new_joint_vecs")
        pos_path = os.path.join(base_path, "new_joints")
        return HumanMLVecSample(vec_path, sample_num, is_mirror), HumanMLPosSample(pos_path, sample_num, is_mirror)

    def open_as_animation(self, path):
        sample = HumanMLPosSample(path)
        locations = sample.positions

        offset_anim, sorted_order, sorted_parents = Animation.animation_from_offsets(self.reference_offsets,
                                                                                     HUMANML_PARENTS)
        assert sorted_order[0] == 0  # later we use pelvis location as hard coded 0

        # convert rotations to be relative to offset angle
        return InverseKinematics.animation_from_positions(positions=locations,
                                                          parents=HUMANML_PARENTS,
                                                          offsets=self.reference_offsets,
                                                          ik_iterations=3)

    @staticmethod
    def get_texts(sample_num):
        sample_id = "%06d" % sample_num
        text_file_name = sample_id + ".txt"
        text_path = os.path.join(DATASET_BASE_PATH, "texts", text_file_name)
        with open(text_path, 'r') as f:
            lines = f.readlines()
        return [line.split('#')[0] for line in lines]

    # def open_as_animation(self, samples):
    #     vec_sample, pos_sample = samples
    #     rotations = Quaternions(np.array([frame.get_rot_data_quat()
    #                                       for frame in vec_sample]))
    #     positions = np.array([self.pos_to_offsets(frame.reference_pos)
    #                           for frame in vec_sample])
    #     orients = self.reference_rot
    #     offsets = self.reference_offsets
    #     return Animation.Animation(rotations, positions, orients, offsets, np.array(self.parents))

    def save_as_bvh(self, in_path, out_path):
        new_anim, anim_from_pos_order, anim_from_pos_parents = self.open_as_animation(in_path)
        BVH.save(out_path, new_anim,
                 names=np.array(HUMANML_JOINT_NAMES)[anim_from_pos_order])


class HumanML2OPConversions:
    def __init__(self):
        op2hm_dict = {0: 15, 1: 12, 2: 16, 3: 18, 4: 20, 5: 17, 6: 19, 7: 21, 8: 0, 9: 1, 10: 4, 11: 7, 12: 2, 13: 5,
                      14: 8,
                      15: 9, 16: 6, 17: 13, 18: 14, 19: 3,
                      20: 22}  # note: 7 & 8 are new in (4,10) & (8,11) respectively
        hm2op_dict = {15: 0, 12: 1, 16: 2, 18: 3, 20: 4, 17: 5, 19: 6, 21: 7, 0: 8, 1: 9, 4: 10, 7: 11, 2: 12, 5: 13,
                      8: 14,
                      9: 15, 6: 16, 13: 17, 14: 18, 3: 19, 22: 20, 10: 11, 11: 14}

        human_ml_len = 22
        openpose_len = 20

        self.switch_hands = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                             13: 14, 16: 17, 18: 19, 20: 21, 14: 13, 17: 16, 19: 18, 21: 20, 15: 15, 22: 22}
        self.forward = op2hm_dict
        self.backward = hm2op_dict
        self.src_len = openpose_len
        self.dst_len = human_ml_len

    def openpose_list_to_humanml(self, idxs):
        return [self.forward[idx] for idx in idxs]

    def openpose_tuple_to_humanml(self, idxs):
        return tuple(self.forward[idx] for idx in idxs)

    def openpose_list_dict_vals_to_humanml(self, dic):
        # for skeletal pooling
        new_dic = {}
        for k in dic.keys():
            new_dic[k] = self.openpose_list_to_humanml(dic[k])
        return new_dic

    def reorder_openpose_to_human_ml(self, array):
        return [(array[self.backward[i]] if self.backward[i] < len(array) else None) for i in
                range(self.src_len)]

    def openpose_tuples_to_humanml(self, tuples):
        return [self.openpose_tuple_to_humanml(t) for t in tuples]

    def openpose_tuples_dict_vals_to_humanml(self, dic):
        # for skeletal pooling
        new_dic = {}
        for k in dic.keys():
            new_dic[k] = self.openpose_tuples_to_humanml(dic[k])
        return new_dic

    def openpose_tuple_dict_to_humanml(self, dic):
        new_dic = {}
        for k in dic.keys():
            new_dic[self.openpose_tuple_to_humanml(k)] = self.openpose_tuple_to_humanml(dic[k])
        return new_dic


class HumanMLNewConversions(HumanML2OPConversions):
    def __init__(self):
        super().__init__()
        reordered_joint_names = \
            ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'Spine1',
             'Spine2', 'Spine3', 'Neck', 'Head', 'L_Collar', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Collar',
             'R_Shoulder', 'R_Elbow', 'R_Wrist']
        reordered_parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

        hm2new_dict = {}
        for i, name in enumerate(HUMANML_JOINT_NAMES):
            hm2new_dict[i] = reordered_joint_names.index(name)
        hm2new_dict[22] = 22
        self.forward = hm2new_dict
        new2hm_dict = {}
        for i, name in enumerate(reordered_joint_names):
            new2hm_dict[i] = list(HUMANML_JOINT_NAMES).index(name)
        new2hm_dict[22] = 22
        self.backward = new2hm_dict
        self.src_len = 22
        self.dst_len = 22


if __name__ == '__main__':
    src = {(0, 3): (0, 22), (6, 3): (0, 3), (9, 6): (6, 3), (12, 9): (9, 6),
                           (15, 12): (12, 9), (9, 14): (9, 6), (17, 14): (9, 14), (17, 19): (17, 14),
                           (19, 21): (17, 19), (9, 13): (9, 6), (16, 13): (9, 13), (16, 18): (16, 13),
                           (18, 20): (16, 18), (0, 2): (0, 22), (2, 5): (0, 2), (5, 8): (2, 5), (8, 11): (5, 8),
                           (0, 1): (0, 22), (1, 4): (0, 1), (4, 7): (1, 4), (7, 10): (4, 7)}
    converter = HumanMLNewConversions()
    print(converter.openpose_tuple_dict_to_humanml(src))

    # print(SampleReader.get_texts(0))

    # num = 777
    # suffix = '_' + str(num) + ".bvh"
    # in_path = r"D:\Documents\University\DeepGraphicsWorkshop\git\HumanML3D\HumanML3D\new_joints\000001.npy"
    # out_path = r"D:\Documents\University\DeepGraphicsWorkshop\git\MoDi\Test_data\char2\000001\000001" + suffix
    # SampleReader().save_as_bvh(in_path, out_path)
