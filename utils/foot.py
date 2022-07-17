import torch
from models.kinematics import ForwardKinematicsJoint
import numpy as np


def get_foot_contact(motion_data, use_glob_pos, axis_up, edge_rot_dict_general, foot_names):
    # from utils.data import foot_names
    n_motions, _, _, n_frames = motion_data.shape
    data_dtype = motion_data.dtype
    # compute joint locations
    root_idx = 0
    assert edge_rot_dict_general['parents_with_root'][0] == -1  # fk class must have root at index 0
    offsets = np.insert(edge_rot_dict_general['offsets_no_root'], root_idx, edge_rot_dict_general['offset_root'],
                        axis=0)
    offsets = np.repeat(offsets[np.newaxis], n_motions, axis=0)
    offsets = torch.from_numpy(offsets).to(motion_data.device).type(data_dtype)
    fk = ForwardKinematicsJoint(edge_rot_dict_general['parents_with_root'], offsets)
    motion_for_fk = motion_data.transpose(1,
                                          3)  # samples x features x joints x frames  ==>  samples x frames x joints x features
    if use_glob_pos:
        #  last 'joint' is global position. use only first 3 features out of it.
        glob_pos = motion_for_fk[:, :, -1, :3]
        if 'use_velocity' in edge_rot_dict_general and edge_rot_dict_general['use_velocity']:
            glob_pos = torch.cumsum(glob_pos, dim=1)
        motion_for_fk = motion_for_fk[:, :, :-1]
    else:
        glob_pos = torch.zeros_like(motion_for_fk[:, :, 0, :3])
    joint_location = fk.forward_edge_rot(motion_for_fk, glob_pos)
    # compute foot contact
    names = edge_rot_dict_general['names_with_root']
    n_foot = len(foot_names)
    idx_foot = [np.nonzero(names == foot_name)[0][0] for foot_name in foot_names]
    foot_location = joint_location[:, :, idx_foot]
    foot_up_location = foot_location[..., axis_up]
    shin_len = offsets[0, idx_foot].pow(2).sum(axis=-1).sqrt()
    # we take the average of all foot heights up to 20% percentile as floor height
    percentile_20 = int(n_motions * n_frames * 0.2)
    # foot_location_sorted = torch.sort(foot_up_location.reshape(n_motions*n_frames, n_foot), axis=-1)[0]
    foot_location_sorted = torch.sort(foot_up_location.reshape(-1, n_foot), axis=0)[0]
    floor_height = torch.mean(foot_location_sorted[:percentile_20], axis=0)
    height_threshold = floor_height + 0.2 * shin_len
    foot_contact = (foot_up_location < height_threshold).type(data_dtype)
    foot_contact = torch.ones_like(foot_contact)
    # Euclidian distance between foot 3D loc now and 2 frames ago
    foot_velocity = (foot_location[:, 2:] - foot_location[:, 0:-2]).pow(2).sum(axis=-1).sqrt()
    # if foot velocity is greater than threshold we don't consider it a contact
    velo_thresh = 0.05 * shin_len
    idx_high_velocity = torch.nonzero(foot_velocity >= velo_thresh, as_tuple=True)
    foot_contact[:, 1:-1][idx_high_velocity] = 0

    # change shape back to motion_data shape: samples x frames x joints  ==>  samples x joints x frames
    foot_contact.transpose_(1,2)

    return foot_contact


def get_foot_velo(motion_data, use_glob_pos, axis_up, edge_rot_dict_general):
    from utils.data import foot_names
    motion_data = motion_data * edge_rot_dict_general['std_tensor'][:, :, :motion_data.shape[2]] + edge_rot_dict_general['mean_tensor'][:, :, :motion_data.shape[2]]

    n_motions, _, _, n_frames = motion_data.shape
    data_dtype = motion_data.dtype
    # compute joint locations
    root_idx = 0
    assert edge_rot_dict_general['parents_with_root'][0] == -1  # fk class must have root at index 0
    offsets = np.insert(edge_rot_dict_general['offsets_no_root'], root_idx, edge_rot_dict_general['offset_root'],
                        axis=0)
    offsets = np.repeat(offsets[np.newaxis], n_motions, axis=0)
    offsets = torch.from_numpy(offsets).to(motion_data.device).type(data_dtype)
    fk = ForwardKinematicsJoint(edge_rot_dict_general['parents_with_root'], offsets)
    motion_for_fk = motion_data.transpose(1,
                                          3)  # samples x features x joints x frames  ==>  samples x frames x joints x features
    if use_glob_pos:
        #  last 'joint' is global position. use only first 3 features out of it.
        glob_pos = motion_for_fk[:, :, -1, :3]
        if 'use_velocity' in edge_rot_dict_general and edge_rot_dict_general['use_velocity']:
            glob_pos = torch.cumsum(glob_pos, dim=1)
        motion_for_fk = motion_for_fk[:, :, :-1]
    else:
        glob_pos = torch.zeros_like(motion_for_fk[:, :, 0, :3])
    joint_location = fk.forward_edge_rot(motion_for_fk, glob_pos)
    # compute foot contact
    names = edge_rot_dict_general['names_with_root']
    n_foot = len(foot_names)
    idx_foot = [np.nonzero(names == foot_name)[0][0] for foot_name in foot_names]
    foot_location = joint_location[:, :, idx_foot]
    foot_velocity = (foot_location[:, 1:] - foot_location[:, :-1]).pow(2).sum(axis=-1).sqrt()
    # if foot velocity is greater than threshold we don't consider it a contact
    foot_velocity = foot_velocity.permute(0, 2, 1)

    return foot_velocity