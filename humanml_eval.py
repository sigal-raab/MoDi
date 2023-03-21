from t2m.final_evaluations import *
from dataclasses import dataclass

from Motion.BVH import load as bvh_load
from Motion.Animation import positions_global
from utils.humanml_utils import position_to_humanml
import numpy as np
from os.path import join as pjoin
from glob import glob
from tqdm import tqdm
import os 

def create_test_std_mean(modi_folder_path, out_path):
    first = True
    anim_parts = glob(pjoin(modi_folder_path,'*.bvh'))
    for part_path in tqdm(anim_parts):
        a, nm,_ = bvh_load(pjoin(part_path))
        motion,_,_,_ = position_to_humanml(positions_global(a), nm)

        if first:
            animations=motion
            first = False
        else:
            animations = np.concatenate((animations,motion), axis=0)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.save(pjoin(out_path,'std.npy'),np.std(animations, axis=0))
    np.save(pjoin(out_path,'mean.npy'),np.mean(animations, axis=0))




if __name__ == '__main__':
    # run the evaluation
    evaluation(log_file)

    # # use this to generate mean and std for processed data
    # # place the files in the checkpoints/t2m/Comp_v6_KLD01/meta
    # create_test_std_mean(r"D:\Documents\University\DeepGraphicsWorkshop\data\preprocessed_data_test",
    #                      r"D:\Documents\University\DeepGraphicsWorkshop\data\eval_std_mean")