"""
compute joint locations from data.
Extract bone lengths.
define offsets with fixed angles.
recompute rotations relative to offset angles.
"""

# imports from python
import os
import torch
import argparse
import numpy as np
import copy

# imports from motion-guided-diffusion
from data_loaders.get_data import get_dataset

# imports from Motion github repository
from Motion.Quaternions import Quaternions
from Motion.Animation import animation_from_offsets, Animation, positions_global
from Motion import BVH
from Motion.InverseKinematics import animation_from_positions

SMPL_JOINT_NAMES = [ # use this only if humanml use the topology of smpl
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
        'L_Hand',  # 22
        'R_Hand',  # 23
    ]

def predefined_offsets():
    # offset is the 1st pose in the 1st motion in the dataset


    #######################################################################################
    # For HumanML3D you need to prepare a tensor similar to the following one
    #######################################################################################

    offsets = dict()
    offsets['humanact12'] = np.array([[0., 0., 0.],
           [0.07004722, 0.08965848, 0.01700827],
           [-0.06590635, 0.09190461, -0.00142403],
           [-0.00778638, -0.10942143, 0.02363084],
           [0.0349078, 0.37407005, 0.02867312],
           [-0.03455295, 0.3792888, 0.05338185],
           [0.01603755, -0.12603089, -0.04651947],
           [-0.01878606, 0.3851236, 0.10859716],
           [0.02559297, 0.38311782, 0.1154949],
           [0.00539924, -0.04703045, -0.03476031],
           [0.04157123, 0.08718348, -0.09331653],
           [-0.04266275, 0.0752641, -0.10339964],
           [-0.01172653, -0.21415776, -0.03979842],
           [0.07220449, -0.1301918, -0.00617102],
           [-0.08784838, -0.11968958, -0.01593818],
           [0.02526567, -0.04771224, -0.06300393],
           [0.09593492, 0.0061667, -0.00650124],
           [-0.08343486, 0.00428885, -0.05815312],
           [0.01848353, 0.2582566, -0.0357407],
           [-0.02811025, 0.24879327, -0.04829484],
           [-0.16308032, -0.04024038, -0.18435043],
           [0.2278388, -0.03592977, -0.10984969],
           [-0.05875964, -0.02148843, -0.05863976],
           [0.06617047, -0.03060508, -0.04460379]])

    return offsets


def main():

    data = get_dataset(name='humanml', num_frames=64)

    one_motion = data[0]['inp'].unsqueeze(0)

    #######################################################################################
    # For HumanML3D you need to prepare a tensor similar to the following one
    #######################################################################################
    locations = foo(one_motion) # some code to extract locations out of the humanml format
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]  # true only if humanml are using smpl topology

    if True:  # change to False in order to take the offsets of a new dataset or change the motion on which the offsets are based
        offsets = predefined_offsets()['humanml']
    else:
        loc_for_offsets = locations[0]
        offsets = loc_for_offsets - loc_for_offsets[parents]
        offsets[0] = loc_for_offsets[0]

    offset_anim, sorted_order, sorted_parents = animation_from_offsets(offsets, parents)
    assert sorted_order[0] == 0  # later we use pelvis location as hard coded 0

    # convert rotations to be relative to offset angle
    new_anim, anim_from_pos_order, anim_from_pos_parents = animation_from_positions(positions=locations, parents=parents, offsets=offsets)

    # save bvh files
    BVH.save(os.path.expanduser('~/tmp/anim_offsets.bvh'), offset_anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])
    # sanity check: display bvh of new motion, compared to bvh of original motion with no fixed offsets. make sure they look identical.
    BVH.save(os.path.expanduser('~/tmp/anim_new.bvh'), new_anim, names=np.array(SMPL_JOINT_NAMES)[anim_from_pos_order])
    smpl_anim, anim_from_pos_order, anim_from_pos_parents = animation_from_positions(positions=locations, parents=parents)
    BVH.save(os.path.expanduser('~/tmp/anim_smpl.bvh'), smpl_anim, names=np.array(SMPL_JOINT_NAMES)[anim_from_pos_order])

    #######################################################################################
    # The next stage is to follow the steps in preprocess_edges.py like expand_topology_edges etc.
    #######################################################################################


if __name__ == '__main__':
    main()
