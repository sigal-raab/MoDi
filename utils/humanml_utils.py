import numpy as np
import torch
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_quaternion
from Motion.Animation import Animation
REFERENCE_ANIMATION_PATH = r"D:\Documents\University\DeepGraphicsWorkshop\git\HumanML3D\HumanML3D\new_joint_vecs\000000.npy"
IDENTITY_ROTATION = [1., 0., 0., 0., 1., 0.]

class HumanMLFrame:
    def __init__(self, sample_vec):
        self.data_vec = sample_vec
        self.joints_num = 22

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
        return matrix_to_quaternion(rotation_6d_to_matrix(torch.from_numpy(self.rot_data)))

    @property
    def foot_contact(self):
        return self.data_vec[259:263]

    def get_foot_contact(self):
        """returns the expected torch tensor"""
        return torch.from_numpy(self.foot_contact)


class HumanMLSample:
    def __init__(self, path, sample_num=None):
        """if sample_num is given, path will be regarded as base path"""
        if sample_num is None:
            self.path = path
        else:
            sample_id = "%06d" % sample_num
            self.path = path + sample_id + '.npy'
        self.joints_num = 22
        self.samples_vec = np.load(self.path)
        self._frames_num = self.samples_vec.shape[0]

    def get_frame(self, frame_num):
        return HumanMLFrame(self.samples_vec[frame_num])

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
        return np.concatenate([np.stack([np.zeros_like(self.root_y), self.root_y, np.zeros_like(self.root_y)], axis=1).reshape((-1, 1, 3)),
                               np.reshape(self.samples_vec[:, 4:67], (-1, self.joints_num - 1, 3))], axis=1)

    @property
    def rot_data(self):
        return np.concatenate([np.array(IDENTITY_ROTATION * self.length).reshape((-1, 1, 6)),
                               np.reshape(self.samples_vec[:, 67:193], (-1, self.joints_num - 1, 6))], axis=1)

    def get_rot_data_quat(self):
        """returns the expected torch tensor"""
        return matrix_to_quaternion(rotation_6d_to_matrix(torch.from_numpy(self.rot_data)))

    @property
    def foot_contact(self):
        return self.samples_vec[:, 259:263]

    def get_foot_contact(self):
        """returns the expected torch tensor"""
        return torch.from_numpy(self.foot_contact)


class SampleReader:
    def __init__(self):
        self.reference_sample = HumanMLSample(REFERENCE_ANIMATION_PATH)
        self.reference_rot = self.reference_sample[0].get_rot_data_quat()
        self.reference_ric = self.reference_sample[0].ric_data

    def open_as_animation(self, path):
        sample = HumanMLSample(path)
        # TODO: Verify that this is the correct way to subtract the base pose
        # rotations = torch.tensor([frame.get_rot_data_quat() - self.reference_rot for frame in sample])
        # positions = torch.tensor([frame.ric_data - self.reference_ric for frame in sample])
        rotations = sample.get_rot_data_quat() - self.reference_rot
        positions = sample.ric_data - self.reference_ric
        orients = self.reference_rot
        offsets = self.reference_ric
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        return Animation(rotations, positions, orients, offsets, parents)


class HumanMLConversions:
    op2hm_dict = {0: 15, 1: 12, 2: 16, 3: 18, 4: 20, 5: 17, 6: 19, 7: 21, 8: 0, 9: 1, 10: 4, 11: 7, 12: 2, 13: 5, 14: 8,
                  15: 9, 16: 6, 17: 13, 18: 14, 19: 3, 20: 22}  # note: 7 & 8 are new in (4,10) & (8,11) respectively
    hm2op_dict = {15: 0, 12: 1, 16: 2, 18: 3, 20: 4, 17: 5, 19: 6, 21: 7, 0: 8, 1: 9, 4: 10, 7: 11, 2: 12, 5: 13, 8: 14,
                  9: 15, 6: 16, 13: 17, 14: 18, 3: 19, 22: 20, 10: 11, 11: 14}

    human_ml_len = 22
    openpose_len = 20

    switch_hands = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                    13: 14, 16: 17, 18: 19, 20: 21, 14: 13, 17: 16, 19: 18, 21: 20, 15: 15,  22: 22}

    @classmethod
    def openpose_list_to_humanml(cls, idxs):
        return [cls.op2hm_dict[idx] for idx in idxs]

    @classmethod
    def openpose_tuple_to_humanml(cls, idxs):
        return tuple(cls.op2hm_dict[idx] for idx in idxs)

    @classmethod
    def openpose_list_dict_vals_to_humanml(cls, dic):
        # for skeletal pooling
        new_dic = {}
        for k in dic.keys():
            new_dic[k] = cls.openpose_list_to_humanml(dic[k])
        return new_dic

    @classmethod
    def reorder_openpose_to_human_ml(cls, array):
        return [(array[cls.hm2op_dict[i]] if cls.hm2op_dict[i] < len(array) else None) for i in range(cls.openpose_len)]

    @classmethod
    def openpose_tuples_to_humanml(cls, tuples):
        return [cls.openpose_tuple_to_humanml(t) for t in tuples]

    @classmethod
    def openpose_tuples_dict_vals_to_humanml(cls, dic):
        # for skeletal pooling
        new_dic = {}
        for k in dic.keys():
            new_dic[k] = cls.openpose_tuples_to_humanml(dic[k])
        return new_dic
    @classmethod
    def openpose_tuple_dict_to_humanml(cls, dic):
        new_dic = {}
        for k in dic.keys():
            new_dic[cls.openpose_tuple_to_humanml(k)] = cls.openpose_tuple_to_humanml(dic[k])
        return new_dic
