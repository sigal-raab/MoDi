import numpy as np
import os
import os.path as osp
import cv2
import math
import matplotlib.pyplot as plt
import copy
from typing import Union

from utils.data import expand_topology_joints
from Motion import InverseKinematics as IK
from Motion import Animation
from Motion import BVH
from utils.data import Joint
from utils.data import calc_bone_lengths
from motion_class import StaticData, DynamicData


FIGURE_JOINTS = ['Head', 'Neck', 'RightArm', 'RightForeArm', 'RightHand', 'LeftArm',
                 'LeftForeArm', 'LeftHand', 'Hips', 'RightUpLeg', 'RightLeg',
                 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']


def pose2im_all(all_peaks, H=512, W=512, foot_contact_info=None):
    limbSeq = [[1, 2], [2, 3], [3, 4],                       # right arm
               [1, 5], [5, 6], [6, 7],                       # left arm
               [8, 9], [9, 10], [10, 11],                    # right leg
               [8, 12], [12, 13], [13, 14],                  # left leg
               [1, 0],                                       # head/neck
               [1, 8],                                       # body,
               ]

    limb_colors = [[0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    [100, 0, 255], [150, 0, 255], [200, 0, 255],
                    [255, 0, 255],
                    [100, 100, 100],
                   ]

    joint_colors = [[255, 0, 255], [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0], [0, 0, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    ]

    image = pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W, foot_contact_info=foot_contact_info)
    return image


def stretch(data, H, W):
    """ Stretch the skeletons proportionally to each other """

    # locate body center in 0,0,0
    data -= data[:, 8:9, :, :]

    # find min/max for each motion and each frame and each axis (that is, across all joints)
    mins = data.min(1)
    maxs = data.max(1)
    diffs = maxs-mins

    scale_from = diffs.max()
    scale_to = min(H, W) - 1

    data -= mins[:, np.newaxis]
    assert data.min() == 0
    data /= scale_from
    assert data.max() == 1
    data *= scale_to
    assert data.max() == scale_to

    return data


def pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W,
            _circle=True, _limb=True, imtype=np.uint8, foot_contact_info=None):
    canvas = np.zeros(shape=(H, W, 3))
    canvas.fill(255)

    if _circle:
        for i in range(len(joint_colors)):
            cv2.circle(canvas, (int(all_peaks[i][0]), int(all_peaks[i][1])), 2, joint_colors[i], thickness=2)

    if foot_contact_info:
        for foot_idx in foot_contact_info:
            if foot_contact_info[foot_idx] > 0.5:
                thickness = -1
            else:
                thickness = 2

            cv2.circle(canvas, center=(int(all_peaks[foot_idx][0]), int(all_peaks[foot_idx][1])), radius=20,
                       color=(0, 0, 0), thickness=thickness)

    if _limb:
        stickwidth = 2

        for i in range(len(limbSeq)):
            limb = limbSeq[i]
            cur_canvas = canvas.copy()
            point1_index = limb[0]
            point2_index = limb[1]

            if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
                point1 = all_peaks[point1_index][0:2]
                point2 = all_peaks[point2_index][0:2]
                X = [point1[1], point2[1]]
                Y = [point1[0], point2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                # cv2.line()
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas.astype(imtype)


def motion2fig_1_motion_3_angles(data, H=512, W=512):
    diffs = [data[0, :, i, 0].max() - data[0, :, i, 0].min() for i in range(3)]
    scale = max(diffs)
    n_frames = data.shape[-1]
    n_samples = 5 # how many samples to take from the whole video
    idx = np.linspace(0, n_frames-1, n_samples).round().astype(int)

    fig, axes = plt.subplots(3, n_samples)
    for sample_idx, i in enumerate(idx):
        for angle_idx, d in enumerate([data[0,:,:2,i], data[0,:,::2,i], data[0,:,1:,i]]):
            img = pose2im_all(d, scale, H, W)
            # axes[angle_idx, i].subplot(3, n_samples, 1)
            axes[angle_idx, sample_idx].axis('off')
            axes[angle_idx, sample_idx].imshow(img[::-1,:]) # image y axis is inverted
    return fig


def foot_info(motions_all: DynamicData):
    foot_contact_info = [[{FIGURE_JOINTS.index(foot_name): frame[foot_name] for foot_name in frame if foot_name in FIGURE_JOINTS}
                          for frame in motion]
                         for motion in motions_all.foot_contact()]

    return foot_contact_info


def motion2fig(motions_all: DynamicData, character_name: str,
               height=512, width=512, n_sampled_frames=5, entity='Edge'):
    if not all([joint in motions_all.motion_statics.names for joint in FIGURE_JOINTS]):
        print(f'Visualisation figure generation is configured only for mixamo characters containing joint {FIGURE_JOINTS}')
        return None
    
    n_sampled_motions = motions_all.shape[0]

    if entity == 'Edge':
        anim, names = motions_all[0].to_anim()

        joints = np.zeros((n_sampled_motions,) + anim.shape + (3,))
        for idx, motion_all in enumerate(motions_all):
            anim, _ = motion_all.to_anim()
            joints[idx] = Animation.positions_global(anim)

        names = list(names)
        figure_indexes = [names.index(joint) for joint in FIGURE_JOINTS]
        foot_contact_info = foot_info(motions_all)

        data = joints[..., figure_indexes, :2]  # b x T x J x 4 -> acesss only the xy projection from K
        data = data.transpose(0, 2, 3, 1)  # samples x frames x joints x features ==> samples x joints x features x frames

    else:
        data = motions_all.motion[..., ::2, :, :].numpy()
        data = data.transpose(0, 2, 1, 3)
        foot_contact_info = None

    data = stretch(data, height, width)

    fig, axes = plt.subplots(n_sampled_motions, n_sampled_frames)
    if axes.ndim == 1: # if there is only one motion
        axes = axes[np.newaxis, :]
    max_w, max_h = np.ceil(data.max(axis=(0,1,3))).astype(int)
    for motion_idx in np.arange(n_sampled_motions):
        for frame_idx in np.arange(n_sampled_frames):
            skeleton = data[motion_idx,:,:,frame_idx]
            img = pose2im_all(skeleton, max_h, max_w, foot_contact_info=foot_contact_info[motion_idx][frame_idx] if foot_contact_info else None)
            axes[motion_idx, frame_idx].axis('off')
            try:
                axes[motion_idx, frame_idx].imshow(img[::-1,:]) # image y axis is inverted
            except:
                pass # in some configurations the image cannot be shown
    return fig


def motion2bvh_rot(motions_all: DynamicData, bvh_file_path):
    for idx, motion_all in enumerate(motions_all):
        anim, names = motion_all.to_anim()

        bvh_file_dir = osp.split(bvh_file_path)[0]
        os.makedirs(bvh_file_dir, exist_ok=True)

        bvh_file_name = osp.split(bvh_file_path)[1]
        bvh_file_path_idx = f'{bvh_file_dir}/{idx}_{bvh_file_name}'

        BVH.save(bvh_file_path_idx, anim, names)


# old function to save figures from Joint model.
def motion2bvh_loc(motion_data, bvh_file_path, parents, type=None):
    if isinstance(motion_data, list): # saving sub pyramid motions
        bl_full = calc_bone_lengths(motion_data[-1], parents=Joint.parents_list[-1])
        for i, sub_motion in enumerate(motion_data):
            is_openpose = (sub_motion.shape == motion_data[-1].shape)
            sub_motion = sub_motion[0]  # drop batch dim
            if type == 'sample' or type =='interp-mix-pyramid':  # displaying sub motions
                suffix = '_' + str(sub_motion.shape[0]) + 'x' + str(sub_motion.shape[2])
                sub_bvh_file_path = bvh_file_path.replace('.bvh', suffix+'.bvh')

                # multiply bone length such that bone lengths in all levels are comparable
                n_joints = sub_motion.shape[0]
                if n_joints==1:
                    continue
                bl_sub_motion = calc_bone_lengths(sub_motion[np.newaxis], parents=parents[i])
                bl_mult = bl_full['mean'].mean() / bl_sub_motion['mean'].mean()
                sub_motion = sub_motion * bl_mult

                # repeat frames so the frame number of the sub_motion would be the same as the final one,
                # in order to make the visualization synchronized
                frame_mult = int(motion_data[-1].shape[-1] / sub_motion.shape[-1])
                sub_motion = sub_motion.repeat(frame_mult, axis=2)
                one_motion2bvh(sub_motion, sub_bvh_file_path, parents=parents[i], is_openpose=is_openpose)
            elif type == 'edit':
                sub_bvh_file_path = bvh_file_path.replace('.bvh', '_'+'{:02d}'.format(i)+'.bvh')
                motion2bvh_loc(motion_data, bvh_file_path, parents, type)
            else:
                raise('unsupported type for list manipulation')
    else:
        if motion_data.ndim == 4:
            assert motion_data.shape[0] == 1
            motion_data = motion_data[0]
        one_motion2bvh(motion_data, bvh_file_path, parents=parents, is_openpose=True)


def motion_to_bvh(motions_all: DynamicData, motion_path: str, entity: str, parents: []=None):
    if entity == 'Edge':
        motion2bvh_rot(motions_all, motion_path)
    elif entity == 'Joint':
        motion2bvh_loc(motions_all.motion.numpy().transpose(1, 0, 2), motion_path, parents)


def one_motion2bvh(one_motion_data, bvh_file_path, parents, is_openpose=True, names=None, expand=False):

    # support non-skel-aware motions with 16 joints
    if one_motion_data.shape[0] == 16:
        one_motion_data = one_motion_data[:15]

    one_motion_data = one_motion_data.transpose(2, 0, 1)  # joint, axis, frame  -->   frame, joint, axis

    if expand:
        one_motion_data, parents, names = expand_topology_joints(one_motion_data, is_openpose, parents, names)
    anim, sorted_order, _ = IK.animation_from_positions(one_motion_data, np.array(parents))
    bvh_file_dir = osp.split(bvh_file_path)[0]
    if not osp.exists(bvh_file_dir):
        os.makedirs(bvh_file_dir)

    if names:
        names = names[sorted_order]

    BVH.save(bvh_file_path, anim, names)
