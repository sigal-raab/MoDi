from t2m.final_evaluations import *
from dataclasses import dataclass

from Motion.BVH import load as bvh_load
from Motion.Animation import positions_global
from utils.humanml_utils import position_to_humanml
import numpy as np
from os.path import join as pjoin
from os import mkdir, rmdir

def create_test_data(in_folder_path, out_path, name_file, save_files=True, already_processed=True):
    names = []
    with open(name_file) as f:
        names = f.readlines()

    if not already_processed:
        #TODO: preprocess all npy files without changing length or spliting
        mkdir(pjoin(out_path,'temp'))


    animations = []
    for name in names:
        a, nm,_ = bvh_load(
                    pjoin(processed_folder_path, f'{name}_joints_1_frames_0.bvh')
                    )
        motion,_,_,_ = position_to_humanml(positions_global(a), nm)
        if save_files:
            np.save(pjoin(out_path,name,'.npy'))
        animations.append(motion)

    np.save(pjoin(out_path,'std.npy',np.std(animations))
    np.save(pjoin(out_path,'mean.npy',np.mean(animations))




if __name__ == '__main__':
    evaluation(log_file)

    # args = DummyArgs()

    # # dataset_opt_path = './t2m/checkpoints/kit/Comp_v6_KLD005/opt.txt'
    # dataset_opt_path = './t2m/checkpoints/t2m/Comp_v6_KLD01/opt.txt'
    # eval_motion_loaders = {
    #     ################
    #     ## HumanML3D Dataset##
    #     ################
    #     # 'Comp_v6_KLD01': lambda: get_motion_loader(
    #     #     './checkpoints/t2m/Comp_v6_KLD01/opt.txt',
    #     #     batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
    #     # ),

    #       ################
    #     ## MoDi Dataset##
    #     ################
    #     'MoDi': lambda: get_modi_loader(
    #         './t2m/checkpoints/t2m/Comp_v6_KLD01/opt.txt', # keep this for other options
    #         batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device,
    #         args=args # add dummyy args here
    #     )

    #     ################
    #     ## KIT Dataset##
    #     ################
    #     # 'Comp_v6_KLD005': lambda: get_motion_loader(
    #     #     './checkpoints/kit/Comp_v6_KLD005/opt.txt',
    #     #     batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
    #     # ),
    # }

    # device_id = 0
    # device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    # #torch.cuda.set_device(device_id)

    # mm_num_samples = 100
    # # mm_num_samples = 0

    # mm_num_repeats = 30
    # mm_num_times = 10

    # diversity_times = 300
    # replication_times = 20
    # batch_size = 32


    # # mm_num_samples = 100
    # mm_num_repeats = 1
    
    # # batch_size = 1

    # gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    # wrapper_opt = get_opt(dataset_opt_path, device)
    # eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # log_file = args.out_path if args.out_path!='' else './t2m_evaluation.log'

    # MoDiTests(eval_motion_loaders,mm_num_samples,mm_num_repeats,mm_num_times,diversity_times,replication_times,batch_size,gt_loader,gt_dataset,wrapper_opt,log_file)
    
    # # animation_4_user_study('./user_study_t2m/')