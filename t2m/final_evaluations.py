from datetime import datetime
import numpy as np
import torch
from t2m.motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from t2m.motion_loaders.model_motion_loaders import get_modi_loader
# from t2m.motion_loaders.model_motion_loaders import get_motion_loader, get_modi_loader
from t2m.t2m_utils.get_opt import get_opt
from t2m.t2m_utils.metrics import *
from t2m.networks.evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from t2m.t2m_utils.plot_script import *
from t2m.scripts.motion_process import *
from t2m.t2m_utils import paramUtil
from t2m.t2m_utils.utils import *



from dataclasses import dataclass

from os.path import join as pjoin


###### idk globals

@dataclass
class DummyArgs:
    # TODO: Change these
    ckpt = r"D:\Documents\University\DeepGraphicsWorkshop\outputs\train_stddev_finetune_from_30k\checkpoint\091999.pt"
    path = r"D:\Documents\University\DeepGraphicsWorkshop\data\preprocessed_data_eval\edge_rot_joints_1_frames_64.npy"
    # ckpt = r"/content/drive/MyDrive/MoDi/chk/077999.pt"
    # path = r"/content/drive/MyDrive/MoDi/MoDi/examples/preprocessed_data_small/edge_rot_joints_1_frames_64.npy"
    out_path = ''
    cfg = None

    std_dev = 0.0510

    device_id = 0
    device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')

    motions = 1 # for 1 motion for each sentence
    criteria = 'torch.nn.MSELoss()'
    truncation = 1
    truncation_mean = 4096
    simple_idx = 0
    sample_seeds = None
    no_idle = False
    return_sub_motions = False
    type = 'Edge'
    dataset = 'humanml'
    batch_size = 256

args = DummyArgs()

# dataset_opt_path = './t2m/checkpoints/kit/Comp_v6_KLD005/opt.txt'
dataset_opt_path = './t2m/checkpoints/t2m/Comp_v6_KLD01/opt.txt'
eval_motion_loaders = {
    ################
    ## HumanML3D Dataset##
    ################
    # 'Comp_v6_KLD01': lambda: get_motion_loader(
    #     './checkpoints/t2m/Comp_v6_KLD01/opt.txt',
    #     batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
    # ),
      ################
    ## MoDi Dataset##
    ################
    'MoDi': lambda: get_modi_loader(
        './t2m/checkpoints/t2m/Comp_v6_KLD01/opt.txt', # keep this for other options
        batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device,
        args=DummyArgs()  # add dummy args here
    )
    ################
    ## KIT Dataset##
    ################
    # 'Comp_v6_KLD005': lambda: get_motion_loader(
    #     './checkpoints/kit/Comp_v6_KLD005/opt.txt',
    #     batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
    # ),
}
device_id = 0
device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
#torch.cuda.set_device(device_id)
mm_num_samples = 4
# mm_num_samples = 0
mm_num_repeats = 20  # should be mm_num_repeats > mm_num_times
mm_num_times = 10  # should be mm_num_repeats > mm_num_times
# diversity_times = 300
diversity_times = 3
replication_times = 1
batch_size = 256
# batch_size = 1

gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
log_file = args.out_path if args.out_path!='' else './t2m_evaluation.log'

########
    
    # animation_4_user_study('./user_study_t2m/')
def plot_t2m(data, save_dir, captions):
    data = gt_dataset.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), wrapper_opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
        # print(ep_curve.shape)

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                # TODO: replace with 
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)



def animation_4_user_study(save_dir):
    motion_loaders = {}
    mm_motion_loaders = {}
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
        motion_loader, mm_motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader
        mm_motion_loaders[motion_loader_name] = mm_motion_loader
    motion_loaders['ground_truth'] = gt_loader
    for motion_loader_name, motion_loader in motion_loaders.items():
        for idx, batch in enumerate(motion_loader):
            if idx > 20:
                break
            word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens = batch
            motions = motions[:, :m_lens[0]]
            # plot_t2m(motions.cpu().numpy(), save_path, captions)
            print('-----%d-----'%idx)
            print(captions)
            print(tokens)
            print(sent_lens)
            print(m_lens)
            ani_save_path = pjoin(save_dir, 'animation', '%02d'%(idx))
            joint_save_path = pjoin(save_dir, 'keypoints', '%02d'%(idx))
            os.makedirs(ani_save_path, exist_ok=True)
            os.makedirs(joint_save_path, exist_ok=True)

            data = gt_dataset.inv_transform(motions[0])
            # print(ep_curves.shape)
            joint = recover_from_ric(data.float(), wrapper_opt.joints_num).cpu().numpy()
            joint = motion_temporal_filter(joint)
            np.save(pjoin(joint_save_path, motion_loader_name+'.npy'), joint)
            # save_path = pjoin(save_dir, '%02d.mp4' % (idx))
            plot_3d_motion(pjoin(ani_save_path, '%s.mp4' % (motion_loader_name)),
                           paramUtil.t2m_kinematic_chain, joint, title=captions[0], fps=20)



def MoDiTests(eval_motion_loaders = {},mm_num_samples = 100,mm_num_repeats = 30,mm_num_times = 10,diversity_times = 300,replication_times = 20,batch_size = 32,gt_loader = None,gt_dataset = None,wrapper_opt = None,log_file = './t2m_evaluation.log'):
    eval_motion_loaders = {}
    mm_num_samples = mm_num_samples
    mm_num_repeats = mm_num_repeats
    mm_num_times = mm_num_times
    diversity_times = diversity_times
    replication_times = replication_times
    batch_size = batch_size
    gt_loader = gt_loader
    gt_dataset = gt_dataset
    wrapper_opt = wrapper_opt
    log_file = log_file
    evaluation(log_file)




if __name__ == '__main__':

    args = DummyArgs()

    # dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    dataset_opt_path = './t2m/checkpoints/t2m/Comp_v6_KLD01/opt.txt'
    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        # 'Comp_v6_KLD01': lambda: get_motion_loader(
        #     './checkpoints/t2m/Comp_v6_KLD01/opt.txt',
        #     batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
        # ),

          ################
        ## MoDi Dataset##
        ################
        'MoDi': lambda: get_modi_loader(
            './t2m/checkpoints/t2m/Comp_v6_KLD01/opt.txt', # keep this for other options
            batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device,
            args=DummyArgs() # add dummyy args here
        )

        ################
        ## KIT Dataset##
        ################
        # 'Comp_v6_KLD005': lambda: get_motion_loader(
        #     './checkpoints/kit/Comp_v6_KLD005/opt.txt',
        #     batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
        # ),
    }

    device_id = 0
    device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    #torch.cuda.set_device(device_id)

    mm_num_samples = 100
    # mm_num_samples = 0

    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 300
    replication_times = 20
    batch_size = 32


    # mm_num_samples = 100
    mm_num_repeats = 1
    
    # batch_size = 1

    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = args.out_path if args.out_path!='' else './t2m_evaluation.log'
    evaluation(log_file)
    # animation_4_user_study('./user_study_t2m/')