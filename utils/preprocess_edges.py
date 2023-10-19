import os
import os.path as osp
import copy
import numpy as np
import argparse

from Motion import BVH
from Motion.Quaternions import Quaternions
from Motion.Animation import transforms_global, positions_global, Animation
from Motion.AnimationStructure import children_list, get_sorted_order
from motion_class import StaticConfig
from utils.data import expand_topology_edges

np.set_printoptions(suppress=True)

DEBUG = False
SAVE_SUB_RESULTS = False


def fix_joint_names(char, names):
    if char == 'Ty':
        names = np.char.lstrip(names, 'Boy:')
    elif char == 'Swat':
        names = np.char.lstrip(names, 'swat:')
    elif char == 'BigVegas':
        names = np.char.lstrip(names, 'newVegas:')
    elif char in ['Andromeda', 'Douglas', 'Jasper', 'Liam', 'Malcolm', 'Pearl', 'Remy', 'Stefani', 'maw']:
        names = np.char.lstrip(names, 'mixamorig:')

    return names


def clean(dir_list, dir):
    remove = []
    for f_name in dir_list:
        if not osp.isdir(osp.join(dir, f_name)):
            remove.append(f_name) # cannot actually remove while iterating the list, because it confuses the iteration
    for r in remove:
        dir_list.remove(r)
    return dir_list

def dilute_joints_anim(anim, req_joint_idx, offset_len_mean):
    # we do not want to change inputs that are given as views
    anim = copy.deepcopy(anim)

    for idx in req_joint_idx[1:]:

        # angle should be changed to the multiplication of angles between idx to nearest ancestor in 'req'
        child = idx
        parent = anim.parents[child]
        offset = anim.offsets[child]
        position = anim.positions[:, child]
        if parent not in np.append(req_joint_idx, [-1]):
            while parent not in np.append(req_joint_idx, [-1]):
                assert idx == 4  # as long as we are using the character Jasper only...
                child = parent
                parent = anim.parents[child]
                offset = offset + anim.offsets[child]
                position = position + anim.positions[:, child]

            highest_child = child
            offset_norm = np.linalg.norm(offset) + 1e-20
            normed_offsets = offset / offset_norm
            assert np.allclose(anim.offsets[idx], anim.positions[:, idx])
            anim.offsets[idx] = normed_offsets * offset_len_mean[idx]
            anim.positions[:, idx] = anim.offsets[idx]
            anim.parents[idx] = parent

            # compute angle using IK
            anim_transforms = transforms_global(anim) # need to compute in the inside of the loop because there changes in the loop to anim
            anim_positions = anim_transforms[:, :, :3, 3]  # look at the translation vector inside a rotation matrix
            anim_rotations = Quaternions.from_transforms(anim_transforms)
            tgt_dirs = anim_positions[:, idx] - anim_positions[:, parent]
            tgt_sums = np.linalg.norm(tgt_dirs, axis=-1) + 1e-20
            tgt_dirs = tgt_dirs / tgt_sums[:, np.newaxis]
            assert np.allclose(tgt_sums, offset_len_mean[idx]) and np.allclose(np.linalg.norm(tgt_dirs, axis=-1), 1)

            src_dirs = positions_global(anim)[:, highest_child] - positions_global(anim)[:, parent]
            src_sums = np.linalg.norm(src_dirs, axis=-1) + 1e-20
            src_dirs = src_dirs / src_sums[:, np.newaxis]
            assert np.allclose(src_sums, offset_len_mean[highest_child]) and np.allclose(np.linalg.norm(src_dirs, axis=-1), 1)

            angles = np.arccos(np.sum(src_dirs * tgt_dirs, axis=-1).clip(-1, 1))

            axises = np.cross(src_dirs, tgt_dirs)
            assert np.allclose(Quaternions.from_angle_axis(angles, axises) * src_dirs, tgt_dirs)
            # axises = np.cross(tgt_dirs, src_dirs)
            axises = -anim_rotations[:, parent] * axises

            delta_rotation = Quaternions.from_angle_axis(angles, axises)
            anim.rotations[:, parent] *= delta_rotation

    anim = anim[:, req_joint_idx]

    return anim


def dilute_joints_anim_old(anim, req_joint_idx):

    # we do not want to change inputs that are given as views
    anim = copy.deepcopy(anim)

    for idx in req_joint_idx:

        # angle should be changed to the multiplication of angles between idx to nearest ancestor in 'req'
        child = idx
        parent = anim.parents[child]
        parent_rot = anim.rotations[:, parent]
        offset = anim.offsets[child]
        position = anim.positions[:, child]
        while parent not in np.append(req_joint_idx,[-1]):
            assert idx == 4 # as long as we are using the character Jasper only...


            # the correct way to remove joints is to run IK between the FK of the parts to remove and the sum of their offsets.
            # say we want to drop one joint, which is joint #1. then we need something like:
            # true joint global location = rot0*pos1 + rot0*rot1*pos2
            # new offset = offset1 + offset2
            # new_rot0 = IK(new_offset, true_joint_global_location)
            child = parent
            parent = anim.parents[child]
            offset = offset + anim.offsets[child]
            position = position + anim.positions[:, child]
            parent_rot = parent_rot * anim.rotations[:, parent]

        anim.rotations[:, parent] = parent_rot
        anim.offsets[idx] = offset
        anim.parents[idx] = parent
        anim.positions[:, idx] = position

    anim = anim[:, req_joint_idx]

    return anim


def dilute_joints_anim_file(anim_path, names_req_in_input, char, out_suffix, offset_len_mean):
    names_req_in_input = np.array(names_req_in_input)
    anim_input, names_input, frametime = BVH.load(anim_path)
    names_input = np.array(names_input)

    names_input_fixed = fix_joint_names(char, names_input)

    # make sure all requested names exist in the input anim
    assert np.isin(names_req_in_input, names_input_fixed).all()

    # indices where 'requested' appear in 'cur'
    idx_req_in_input = np.nonzero(names_req_in_input[:, None] == names_input_fixed)[1]

    anim_input_exp, idx_req_in_input_exp, names_input_exp, offset_len_mean_exp = \
        expand_topology_edges(anim_input, idx_req_in_input, names_input_fixed, offset_len_mean, nearest_joint_ratio=1)
    if DEBUG:
        name, ext = osp.splitext(anim_path)
        anim_path_expanded = name + '_expanded' + ext
        BVH.save(filename=anim_path_expanded, anim=anim_input_exp, names=names_input_exp, frametime=frametime)
        idx_input_req_in_input_exp = np.nonzero(names_req_in_input[:, None] == names_input_exp)[1] # indices without expanded joints
        sanity_check_joint_loc(anim_input_exp, idx_input_req_in_input_exp, anim_input, idx_req_in_input)

    anim_diluted = dilute_joints_anim(anim_input_exp, idx_req_in_input_exp, offset_len_mean_exp)
    names_diluted = names_input_exp[idx_req_in_input_exp]

    if DEBUG:
        idx_req_in_diluted = np.nonzero(names_req_in_input[:, None] == names_diluted)[1]
        sanity_check_joint_loc(anim_diluted, idx_req_in_diluted, anim_input, idx_req_in_input)

    if SAVE_SUB_RESULTS:
        name, ext = osp.splitext(anim_path)
        anim_path_diluted = name + out_suffix + ext
        BVH.save(filename=anim_path_diluted, anim=anim_diluted, names=names_diluted, frametime=frametime)


def sanity_check_joint_loc(anim1, idx_req1, anim2, idx_req2):
    """ make sure joint locations (for requested joints only) in anim1 and anim2 are the same """
    idx_req1.sort()
    idx_req2.sort()
    assert np.allclose(positions_global(anim2)[:, idx_req2], positions_global(anim1)[:, idx_req1])


def compute_mean_bone_length(animation_names, char_dir):
    one_char_offsets = None
    for anim_name in animation_names:
        anim_path = os.path.join(char_dir, anim_name, anim_name + '.bvh')
        anim, _, _ = BVH.load(anim_path)

        global_location = positions_global(anim)[:, :]
        parents = copy.deepcopy(anim.parents)
        parents[3] = 0
        anim_offsets = (global_location[:, 1:] - global_location[:, parents[1:]])
        anim_offsets = np.concatenate([global_location[:, :1], anim_offsets],
                                      axis=1)  # for root joint (index 0), put the offset from the global zero
        if one_char_offsets is None:
            one_char_offsets = anim_offsets
        else:
            one_char_offsets = np.append(one_char_offsets, anim_offsets, axis=0)
    offset_len_mean = np.linalg.norm(one_char_offsets, axis=2).mean(axis=0)
    offset_len_std = np.linalg.norm(one_char_offsets, axis=2).std(axis=0)

    return offset_len_mean, offset_len_std


def idx_list1_in_list2(list1, list2):
    """ in which indices of list2 do the items of list1 appear
        example: list1 = [1, 3, 7]
                 list2 = [0, 8, 7, 1, 19, 3, 22]
                 idx = [2, 3, 5]
    """
    idx = np.nonzero(list1[:, None] == list2)[1]
    idx.sort()
    return idx


def get_animation_names(char, data_root):
    char_dir = os.path.join(data_root, char)
    animation_names = os.listdir(char_dir)
    animation_names = clean(animation_names, char_dir)
    if DEBUG:
        n_anim = 10
    else:
        n_anim = len(animation_names)
    animation_names = animation_names[:n_anim]
    return animation_names, char_dir


def joint_rot_2_edge_rot(anim, anim_path, out_suffix, names, frametime):
    """ refer rotations to edge itself rather than to parent joint """

    n_joints = anim.shape[1]
    remove = np.array([], dtype=int)
    edge_rot = copy.deepcopy(anim.rotations)
    parents = copy.deepcopy(anim.parents)

    for joint_idx in range(n_joints - 1, 0, -1):  # go from leaf to root
        parent = anim.parents[joint_idx]
        edge_rot[:, joint_idx] = anim.rotations[:, parent]
        while (anim.offsets[parent] == np.zeros(3)).all():
            assert parent not in remove  # a dummy joint should parent only one child
            remove = np.append(remove, parent)
            parent = anim.parents[parent]
            if parent != 0:  # pelvis rotation should remain seperated from bones
                edge_rot[:, joint_idx] *= Quaternions(anim.rotations[:, parent].qs)  # for some reason isinstance(anim.rotations[:, parent], Quaternions) is False
            parents[joint_idx] = parent

    keep = np.setdiff1d(np.arange(1, n_joints), remove)

    # since joints are re-ordered, we need to take spacial care on parents reorder
    keep_plus_0 = np.insert(keep, 0, 0)
    order_inversed = {num: i for i, num in enumerate(keep_plus_0)}
    order_inversed[-1] = -1
    reindexed_parents = np.array([order_inversed[parents[i]] for i in keep_plus_0])

    edge_rot = edge_rot[:, keep]
    names_rot = names[keep_plus_0]
    edge_rot_dict = {'rot_edge_no_root': edge_rot, 'offsets_no_root': anim.offsets[keep],
                          'parents_with_root': reindexed_parents, 'offset_root': anim.offsets[0],
                          'pos_root': anim.positions[:,0], 'rot_root': anim.rotations[:,0], 'names_with_root': names_rot}

    # save results
    if SAVE_SUB_RESULTS:
        name = osp.splitext(anim_path)[0]
        anim_path_edge_rot = name + out_suffix + '_rot.npy'
        np.save(anim_path_edge_rot, edge_rot_dict)

    if DEBUG:
        sanity_check_joint_rot_2_edge_rot(edge_rot_dict, anim, anim_path, frametime)

    return edge_rot_dict


def basic_anim_from_rot(edge_rot_dict, root_name='Hips'):
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

    return anim_edges


def anim_from_edge_rot_dict(edge_rot_dict, root_name='Hips'):
    n_frames = edge_rot_dict['rot_edge_no_root'].shape[0]
    anim_edges = basic_anim_from_rot(edge_rot_dict, root_name)

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


def sanity_check_joint_rot_2_edge_rot(edge_rot_dict, anim_full, anim_path=None, frametime=None):
    ''' sanity check: anim can be re-created '''

    # reconstruct a full animation from edge_rot_dict
    anim_reconstructed = anim_from_edge_rot_dict(edge_rot_dict)[0]

    assert np.allclose(positions_global(anim_full), positions_global(anim_reconstructed))

    # save to disk
    name, suffix = osp.splitext(anim_path)
    sanity_file = name + '_edges_rot_sanity_check' + suffix
    BVH.save(sanity_file, anim_reconstructed, frametime=frametime)


def sanity_check_dilute_joints(anim_diluted, anim_input, idx_req_in_input, names_diluted, names_req_in_input):
    if DEBUG:
        idx_req_in_diluted = np.nonzero(names_req_in_input[:, None] == names_diluted)[1]
        sanity_check_joint_loc(anim_diluted, idx_req_in_diluted, anim_input, idx_req_in_input)


def sanity_check_expand_topology(anim_input, anim_input_exp, anim_path, frametime, idx_req_in_input, names_input_exp,
                                 names_req_in_input):
    if DEBUG:
        name, ext = osp.splitext(anim_path)
        anim_path_expanded = name + '_expanded' + ext
        BVH.save(filename=anim_path_expanded, anim=anim_input_exp, names=names_input_exp, frametime=frametime)
        idx_input_req_in_input_exp = np.nonzero(names_req_in_input[:, None] == names_input_exp)[
            1]  # indices without expanded joints
        sanity_check_joint_loc(anim_input_exp, idx_input_req_in_input_exp, anim_input, idx_req_in_input)


def sanity_check_split(all_clips, anim_joint_rot, clip_len, frametime, mot_dir, divisor):
    for i, clip in enumerate(all_clips):
        anim_joint_rot_clip = anim_joint_rot[i * (clip_len // divisor): i * (clip_len // divisor) + clip_len, :]
        anim_edge_rot_clip, names_clip = anim_from_edge_rot_dict(clip)
        assert np.allclose(positions_global(anim_joint_rot_clip),
                           positions_global(anim_edge_rot_clip))
        sanity_file = osp.join(mot_dir, '{}.bvh'.format(i+1))
        BVH.save(sanity_file, anim_edge_rot_clip, names=names_clip, frametime=frametime)


def split_to_fixed_length_clips(anim_edge_rot, anim_dir_path, out_suffix, clip_len, stride, root_path, anim_joint_rot,
                                frametime):
    assert clip_len % stride == 0
    divisor = clip_len // stride

    mot_dir = os.path.join(anim_dir_path, f'motions{out_suffix}')
    if SAVE_SUB_RESULTS:
        os.makedirs(mot_dir, exist_ok=True)

    n_frames = anim_edge_rot['rot_edge_no_root'].shape[0]
    n_clips = n_frames // (clip_len // divisor) - (divisor - 1)
    all_clips = np.empty(max(n_clips, 0), dtype=dict)
    all_motion_names = list()

    for i in range(n_clips):
        clip = copy.deepcopy(anim_edge_rot)
        frame_from = i * (clip_len // divisor)
        frame_to = i * (clip_len // divisor) + clip_len
        clip['rot_edge_no_root'] = clip['rot_edge_no_root'][frame_from: frame_to, :]
        clip['pos_root'] = clip['pos_root'][frame_from: frame_to, :]
        clip['rot_root'] = clip['rot_root'][frame_from: frame_to]

        save_path = os.path.join(mot_dir, '{}.npy'.format(i + 1))
        if SAVE_SUB_RESULTS:
            np.save(save_path, clip)

        all_clips[i] = clip
        all_motion_names.append(save_path)
        assert clip['pos_root'].shape[0] == clip_len

    all_motion_names = np.array(all_motion_names)
    for file_idx, motion_name in enumerate(all_motion_names):
        all_motion_names[file_idx] = osp.relpath(motion_name, root_path) # remove root_path from name

    if DEBUG:
        sanity_check_split(all_clips, anim_joint_rot, clip_len, frametime, mot_dir, divisor)

    return all_clips, all_motion_names


def preprocess_edge_rot(data_root, out_suffix, dilute_intern_joints, clip_len, stride):
    character_names = os.listdir(data_root)
    character_names = clean(character_names, data_root)

    anim_edges_split_all_chars = None
    names_split_all_chars = np.empty(0)

    for char in character_names:

        static_config = StaticConfig(char.lower())
        requested_joint_names = np.array(static_config['requested_joints_names'])

        animation_names, char_dir = get_animation_names(char, data_root)

        # compute mean bone length for diluted topology, for all motions of this character
        if dilute_intern_joints:
            offset_len_mean, offset_len_std = compute_mean_bone_length(animation_names, char_dir)
        else:
            offset_len_mean = offset_len_std = None

        for anim_idx, anim_name in enumerate(animation_names):
            anim_dir_path = osp.join(char_dir, anim_name)
            anim_file_path = os.path.join(anim_dir_path, anim_name + '.bvh')

            anim_input, names_input, frametime = BVH.load(anim_file_path)
            names_input = np.array(names_input)

            names_input_fixed = fix_joint_names(char, names_input)
            names_req_in_input = requested_joint_names

            # make sure all requested names exist in the input anim
            assert np.isin(requested_joint_names, names_input_fixed).all()

            # indices where 'requested' appear in 'cur'
            idx_req_in_input = idx_list1_in_list2(names_req_in_input, names_input_fixed) #  np.nonzero(requested_joint_names[:, None] == names_input_fixed)[1]

            ###
            # expand topology: add dummy vertices where a joint has more than one child
            ###
            anim_input_exp, idx_req_in_input_exp, names_input_exp, offset_len_mean_exp = \
                expand_topology_edges(anim_input, idx_req_in_input, names_input_fixed, offset_len_mean, nearest_joint_ratio=1)
            sanity_check_expand_topology(anim_input, anim_input_exp, anim_file_path, frametime, idx_req_in_input,
                                         names_input_exp, names_req_in_input)

            ###
            # remove joints that are not required for the algorithm
            ###
            anim_diluted = dilute_joints_anim(anim_input_exp, idx_req_in_input_exp, offset_len_mean_exp)
            names_diluted = names_input_exp[idx_req_in_input_exp]
            sanity_check_dilute_joints(anim_diluted, anim_input, idx_req_in_input, names_diluted, names_req_in_input)
            name, ext = osp.splitext(anim_file_path)
            if SAVE_SUB_RESULTS:
                anim_path_diluted = name + out_suffix + ext
                BVH.save(filename=anim_path_diluted, anim=anim_diluted, names=names_diluted, frametime=frametime)

            ###
            # convert joint rotations to edge rotations
            ###
            anim_edge_rot = joint_rot_2_edge_rot(anim_diluted, anim_file_path, out_suffix, names_diluted, frametime)

            ###
            # split the clip to a set of fixed length clips
            ###
            anim_edges_split, file_names = \
                split_to_fixed_length_clips(anim_edge_rot, anim_dir_path, out_suffix, clip_len=clip_len, stride=stride,
                                            root_path=data_root, anim_joint_rot=anim_diluted, frametime=frametime)

            if anim_edges_split_all_chars is None:
                anim_edges_split_all_chars = anim_edges_split
            else:
                anim_edges_split_all_chars = np.append(anim_edges_split_all_chars, anim_edges_split)
            names_split_all_chars = np.append(names_split_all_chars, file_names)

            if anim_idx % 100 == 0:
                print('{}/{} Done'.format(anim_idx, len(animation_names)))
    ###
    # save final files
    ###
    anim_edges_split_all_chars_file_name = osp.join(data_root, f'edge_rot_{out_suffix}.npy')
    np.save(anim_edges_split_all_chars_file_name, anim_edges_split_all_chars)
    names_split_all_chars_file_name = osp.join(data_root, f'file_names_{out_suffix}.npy')
    np.savetxt(names_split_all_chars_file_name, names_split_all_chars, fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preproces bvh files such that they can be used as input motions with edge rotation data.")

    parser.add_argument("--data_root", type=str, help="Base folder for all characters.", required=True)
    parser.add_argument("--out_suffix", type=str, help="Suffix for created files.")
    parser.add_argument("--clip_len", type=int, default=64, help="Length (in frames) of sub-clips to be used by the network.")
    parser.add_argument("--stride", type=int, default=32, help="Stride size between the beginning of one clip to the beginning of the following one.")
    args = parser.parse_args()

    if args.out_suffix is None:
        args.out_suffix = '_joints_1_frames_' + str(args.clip_len)
    preprocess_edge_rot(args.data_root, args.out_suffix, dilute_intern_joints=False, clip_len=args.clip_len, stride=args.stride)
