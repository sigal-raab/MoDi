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


foot_names = ['LeftFoot', 'RightFoot']

class openpose_joints(object):
    def __init__(self):
        super().__init__()

        self.oredered_joint_names = \
            np.array(['chin',  'collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
                      'pelvis', 'r_heap', 'r_knee', 'r_ankle', 'l_heap', 'l_knee', 'l_ankle'])
        self.parent_joint_names = \
            np.array(['collar', 'pelvis',  'collar', 'r_shoulder', 'r_elbow', 'collar', 'l_shoulder', 'l_elbow',
                      np.nan, 'pelvis', 'r_heap', 'r_knee', 'pelvis', 'l_heap', 'l_knee'])

    def name2idx(self, joint_names):
        multiple_names =  isinstance(joint_names, np.ndarray)
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


class Joint:
    n_channels = 3

    @staticmethod
    def str():
        return 'Joint'

    skeletal_pooling_dist_1 = [{0:[0, 1]},
                           # root: [collar,pelvis]
                           {0:[0,1,2,3],                             1:[0,3,4,5]},
                           # collar:[collar,r_wrist,l_wrist,pelvis], pelvis:[pelvis,collar,r_ankle,l_ankle]
                           {0:[0,1,3,5], 1:[1,2], 2:[3,4], 3:[0,5,6,8], 4:[6,7], 5:[8,9]},
                           {0:[0,1,2,5], 1:[2,3], 2:[3,4], 3:[5,6], 4:[6,7], 5:[8,9,12], 6:[9,10], 7:[10,11], 8:[12,13], 9:[13,14]}]
    skeletal_pooling_dist_0 = [{0:[1]},
                           {0:[0], 1:[3]},
                           {0:[0], 1:[2], 2:[4], 3:[5], 4:[7], 5:[9]},
                           {0:[1], 1:[3], 2:[4], 3:[6], 4:[7], 5:[8], 6:[10], 7:[11], 8:[13], 9:[14]}]
    oj = openpose_joints()
    parents_list = [[-1],
               [-1,0],
               [3,0,0,-1,3,3],
               [5,0,1,0,3,-1,5,6,5,8],
               list(oj.name2idx(oj.parent_joint_names))]  #full skeleton possesses openpose topology


def collect_motions_loc(root_path):
    """ Read mixamo npy files from disk and concatente to one single file. Save together with a file of motion paths (character / motion_type / sub_motion_idx)"""
    npy_files = glob(osp.join(root_path, '*', '*', 'motions', '*.npy'))

    motion_shape = np.load(npy_files[0]).shape # just to get motion shape
    assert motion_shape[0] == 15
    motion_shape = tuple(np.array(motion_shape)+np.array([1,0,0])) # make sure the number of joints is an exponent of 2
    all_motions = np.zeros((len(npy_files),)+motion_shape)
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

    if motion_data.ndim == 3: # no batch dim
        motion_data = motion_data[np.newaxis]

    # support 16 joints skeleton
    if motion_data.shape[1] == 16:
        motion_data = motion_data[:,:15]

    n_joints = motion_data.shape[1]
    if parents is None or names is None:
        opj = openpose_joints()
        is_openpose = n_joints in [len(opj.oredered_joint_names), len(opj.oredered_joint_names)+1]
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

    bone_lengths = pd.DataFrame(index=names, columns=['mean','std'])
    for joint_idx in range(n_joints):
        parent_idx = parents[joint_idx]
        joint_name = names[joint_idx]
        if parent_idx not in [None, -1]:
            all_bone_lengths = np.linalg.norm(motion_data[:, joint_idx, :, :] - motion_data[:, parent_idx, :, :], axis=1)
            bone_lengths['mean'][joint_name] = all_bone_lengths.mean()
            bone_lengths['std'][joint_name]  = all_bone_lengths.std()
        else:
            bone_lengths.drop(joint_name, inplace=True)

    return bone_lengths


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
    parent_req = np.zeros(n_joints_all) # fixed parents according to req
    n_children_req = np.zeros(n_joints_all) # number of children per joint in the fixed topology
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
    n_super_children = n_children_req[super_parents].sum().astype(int) # total num of multiple children

    if n_super_children == 0:
        return anim, req_joint_idx, names, offset_len_mean  # can happen in lower hierarchy levels

    # prepare space for expanded joints, at the end of each array
    anim.offsets = np.append(anim.offsets, np.zeros(shape=(n_super_children,3)), axis=0)
    anim.positions = np.append(anim.positions, np.zeros(shape=(n_frames, n_super_children,3)), axis=1)
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
            if child in req_joint_idx:
                anim.parents[new_joint_idx] = parent
                anim.parents[child] = new_joint_idx
                names[new_joint_idx] = names[parent]+'_'+names[child]

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


def expand_topology_joints(one_motion_data, is_openpose, parents=None, names = None):
    # one_motion_data: frame x joint x axis
    if is_openpose:
        one_motion_data, parents, names = expand_topology_joints_openpose(one_motion_data)
    else:
        one_motion_data, parents, names = expand_topology_joints_general(one_motion_data, parents, names)

    return one_motion_data, parents, names


def expand_topology_joints_general(one_motion_data, parents, names = None, nearest_joint_ratio=0.9):
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


def to_cpu(motion_data):
    if motion_data is None:
        return None
    if not isinstance(motion_data, list):
        motion_data = motion_data.cpu()
    else:
        for i, motion in enumerate(motion_data):
            motion_data[i] = to_cpu(motion)
    return motion_data


def motion_from_raw(args, motion_data_raw, motion_statics):
    if args.entity == 'Joint':
        if args.skeleton:
            # motion data has an additional dummy joint to support prev architecture. we don't need it anymore so we drop it
            motion_data = motion_data_raw[:, :15]
        else:
            motion_data = motion_data_raw
        # compute mean, maybe add a motion 'Joint' like in 2D-Motion-Retargeting?
        motion_data -= motion_data[:, 8:9, :, :] # locate body center at 0,0,0
        if args.normalize:
            mean_joints = motion_data.mean(axis=(0,3))
            mean_joints = mean_joints[np.newaxis, :, :, np.newaxis]
            std_joints = motion_data.std(axis=(0,3))
            std_joints = std_joints[np.newaxis, :, :, np.newaxis]
            std_joints[np.where(std_joints < 1e-9)] = 1e-9
            assert (std_joints >= 1e-9).all()
            motion_data = (motion_data - mean_joints) / std_joints
        else:
            mean_joints = np.zeros((1, motion_data.shape[1], motion_data.shape[2], 1))
            std_joints = np.ones_like(mean_joints)
    else:  # entity == 'Edge'
        edge_rot_dicts = copy.deepcopy(motion_data_raw)
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
            if True: # to be changed to "if quaternions" once using other rotation representations (6D, euler...)
                # turn 3D pose into 4D by padding with zeros, to match quaternion dims
                dim_delta = motion_data.shape[-1] - root_pos_data.shape[-1]
                root_pos_data = np.concatenate([root_pos_data, np.zeros(root_pos_data.shape[:-1] + (dim_delta,))],
                                               axis=2)

            motion_data = np.concatenate([motion_data, root_pos_data[:,:,np.newaxis]], axis=2)

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
        normalisation_data = {'mean':  mean_joints.transpose(0, 2, 1, 3), 'std': std_joints.transpose(0, 2, 1, 3)}

        if args.foot:
            motion_data_torch = torch.from_numpy(motion_data).transpose(1, 2)
            motion_data_torch = append_foot_contact(motion_data_torch, motion_statics, normalisation_data,
                                                    args.glob_pos, args.use_velocity, args.axis_up)
            motion_data = motion_data_torch.transpose(1, 2).numpy()

            # Do not normalize the contact label
            padding = np.zeros((1, motion_data.shape[1] - mean_joints.shape[1], motion_data.shape[2], 1))
            mean_joints = np.append(mean_joints, padding, axis=1)
            std_joints = np.append(std_joints, np.ones_like(padding), axis=1)

    normalisation_data = {'mean':  mean_joints.transpose(0, 2, 1, 3), 'std': std_joints.transpose(0, 2, 1, 3)}

    return motion_data, normalisation_data


def append_foot_contact(motion_data, motion_statics, normalisation_data, global_position, use_velocity, axis_up):
    foot_contact = get_foot_contact(motion_data, motion_statics, normalisation_data,
                                    use_global_position=global_position, use_velocity=use_velocity, axis_up=axis_up)

    #  pad foot_contact to the size of motion_data features (quaternions or other)
    foot_contact_padded = torch.zeros_like(motion_data[:, :, :motion_statics.foot_number])
    foot_contact_padded[:, 0] = foot_contact

    # concatenate on edges axis (foot_contact is like an extra edge)
    motion_data = torch.cat((motion_data, foot_contact_padded), dim=2)

    return motion_data


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        # refrain from computing gradients for parameters that are 'non_grad': masks, etc.
        if flag == False or \
                flag == True and hasattr(model, 'non_grad_params') and name not in model.non_grad_params:
            p.requires_grad = flag


if __name__ == "__main__":
    pass
