from evaluation.models.stgcn import STGCN
import re
import pandas as pd
import os.path as osp
import os
import datetime

import torch

from torch.utils.data import DataLoader
import numpy as np
from utils.data import anim_from_edge_rot_dict, un_normalize, edge_rot_dict_from_edge_motion_data, motion_from_raw
from utils.data import Joint, Edge # to be used in 'eval'
import sys as _sys
from evaluation.action2motion.fid import calculate_fid
from evaluation.action2motion.diversity import calculate_diversity
from evaluation.metrics.kid import calculate_kid
from evaluation.metrics.precision_recall import precision_and_recall
from Motion import Animation
from matplotlib import pyplot as plt
from generate import get_gen_mot_np, sample
from utils.pre_run import EvaluateOptions, load_all_form_checkpoint
from utils.humanml_utils import position_to_humanml

TEST = False


def generate(args, g_ema, device, mean_joints, std_joints, entity, texts=None, edge_rot_dict_general=None):

    # arguments required by generation
    args.sample_seeds = None
    args.no_idle = False
    args.return_sub_motions = False
    args.truncation = 1
    args.truncation_mean = 4096

    args.motions = 2000
    if TEST:
        args.motions = 64

    with torch.no_grad():
        g_ema.eval()
        mean_latent = g_ema.mean_latent(args.truncation_mean)
        args.truncation = 1
        generated_motions, texts = sample(args, g_ema, device, mean_latent, texts)

    generated_motion = generated_motions.motion
    # convert motion to numpy
    if isinstance(generated_motion, pd.Series): # yes
        index = generated_motion.index
        if not isinstance(generated_motion.iloc[0], list) and \
                generated_motion.iloc[0].ndim == 4 and generated_motion.iloc[0].shape[0] > 1:
            generated_motion = generated_motion.apply(lambda motions: torch.unsqueeze(motions, 1)) # add a batch dimension
            generated_motion = generated_motion.apply(list) # get_gen_mot_np expects lists
        generated_motion = generated_motion.tolist()
    else:
        assert isinstance(generated_motion, list)
        index = range(len(generated_motion))

    generated_motion_np, _ = get_gen_mot_np(args, generated_motion, mean_joints, std_joints)
    generated_motions = np.concatenate(generated_motion_np, axis=0)

    if entity.str() == 'Joint':
        return generated_motions, texts

    if edge_rot_dict_general is None:
        _, _, _, edge_rot_dict_general = motion_from_raw(args, np.load(args.path, allow_pickle=True))
    generated_motions = convert_motions_to_location(args, generated_motion_np, edge_rot_dict_general)
    return generated_motions, texts


def convert_motions_to_location(args, generated_motion_np, edge_rot_dict_general):
    edge_rot_dict_general['std_tensor'] = edge_rot_dict_general['std_tensor'].cpu()
    edge_rot_dict_general['mean_tensor'] = edge_rot_dict_general['mean_tensor'].cpu()
    if args.dataset == 'mixamo':
        edge_rot_dict_general['offsets_no_root'] /= 100 ## not needed in humanact

    root_name = 'Hips'
    if args.dataset == 'humanml':
        root_name = 'Pelvis'

    generated_motions = []

    # get anim for xyz positions
    motion_data = un_normalize(generated_motion_np, mean=edge_rot_dict_general['mean'].transpose(0, 2, 1, 3), std=edge_rot_dict_general['std'].transpose(0, 2, 1, 3))
    anim_dicts, frame_mults, is_sub_motion = edge_rot_dict_from_edge_motion_data(motion_data, type='sample', edge_rot_dict_general = edge_rot_dict_general)

    for j, (anim_dict, frame_mult) in enumerate(zip(anim_dicts, frame_mults)):
        anim, names = anim_from_edge_rot_dict(anim_dict, root_name=root_name)
        # compute global positions using anim
        positions = Animation.positions_global(anim)

        if args.dataset=='humanml':
            positions,_,_,_ = position_to_humanml(positions,names)
            generated_motions.append(positions)

        else:
            # sample joints relevant to 15 joints skeleton
            positions_15joints = positions[:, [7, 6, 15, 16, 17, 10, 11, 12, 0, 23, 24, 25, 19, 20, 21]] # openpose order R then L
            positions_15joints = positions_15joints.transpose(1, 2, 0)
            positions_15joints_oriented = positions_15joints.copy()
            if args.dataset == 'mixamo':
                positions_15joints_oriented = positions_15joints_oriented[:, [0, 2, 1]]
                positions_15joints_oriented[:, 1, :] = -1 * positions_15joints_oriented[:, 1, :]
            generated_motions.append(positions_15joints_oriented)

    generated_motions = np.asarray(generated_motions)
    return generated_motions

#endregion

def calculate_activation_statistics(activations):
    activations = activations.cpu().detach().numpy()
    # activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def initialize_model(device, modelpath, dataset='mixamo'):
    if dataset == 'mixamo':
        num_classes = 15
    elif dataset == 'humanact12':
        num_classes = 12
    model = STGCN(in_channels=3,
                  num_class=num_classes,
                  graph_args={"layout": 'openpose', "strategy": "spatial"},
                  edge_importance_weighting=True,
                  device=device)
    model = model.to(device)
    state_dict = torch.load(modelpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_features(model, iterator):
    device = 'cuda'
    activations = []
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
        # for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            batch_for_model = {}
            batch_for_model['x'] = batch.to(device).float()
            model(batch_for_model)
            activations.append(batch_for_model['features'])
            predictions.append(batch_for_model['yhat'])
            # labels.append(batch_for_model['y'])
        activations = torch.cat(activations, dim=0)
        predictions = torch.cat(predictions, dim=0)
        # labels = torch.cat(labels, dim=0)
        # shape torch.Size([16, 15, 3, 64]) (batch, joints, xyz, frames)
    return activations, predictions


def main(args_not_parsed):
    parser = EvaluateOptions()
    args = parser.parse_args(args_not_parsed)
    device = args.device

    g_ema, discriminator, checkpoint, entity, mean_joints, std_joints = load_all_form_checkpoint(args.ckpt, args)

    if not (getattr(args, 'test_model', None) ^ getattr(args, 'test_actor', None)):
        setattr(args, 'test_model', True)
        setattr(args, 'test_actor', False)
    if args.act_rec_gt_path is None:
        args.act_rec_gt_path = args.path

    modelpath = "evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"
    if args.dataset == 'humanact12':
        modelpath = 'evaluation/humanact12_checkpoint_0150_acc_1.0.pth.tar'

    # initialize model
    model = initialize_model(device, modelpath, args.dataset)

    if args.test_model:
        # generate motions
        generated_motions, texts = generate(args, g_ema, device, mean_joints, std_joints, entity=entity)
        generated_motions = generated_motions[:, :15]

    elif args.test_actor:
        actor_res = np.load(args.actor_motions_path, allow_pickle=True)
        generated_motions = actor_res[:, :15]

    generated_motions -= generated_motions[:, 8:9, :, :]  # locate root joint of all frames at origin

    iterator_generated = DataLoader(generated_motions, batch_size=64, shuffle=False, num_workers=8)

    # compute features of generated motions
    generated_features, generated_predictions = compute_features(model, iterator_generated)
    generated_stats = calculate_activation_statistics(generated_features)


    # dataset motions
    motion_data_raw = np.load(args.act_rec_gt_path, allow_pickle=True)
    motion_data = motion_data_raw[:, :15]
    motion_data -= motion_data[:, 8:9, :, :]  # locate root joint of all frames at origin
    iterator_dataset = DataLoader(motion_data, batch_size=64, shuffle=False, num_workers=8)

    # compute features of dataset motions
    dataset_features, dataset_predictions = compute_features(model, iterator_dataset)
    real_stats = calculate_activation_statistics(dataset_features)

    print(f"evaluation resutls for model {args.ckpt}\n")

    fid = calculate_fid(generated_stats, real_stats)
    print(f"FID score: {fid}\n")

    print("calculating KID...")
    kid = calculate_kid(dataset_features.cpu(), generated_features.cpu())
    (m, s) = kid
    print('KID : %.3f (%.3f)' % (m, s))
    print()

    dataset_diversity = calculate_diversity(dataset_features)
    generated_diversity = calculate_diversity(generated_features)
    print(f"Diversity of generated motions: {generated_diversity}")
    print(f"Diversity of dataset motions: {dataset_diversity}\n")

    if args.fast:
        print("Skipping precision-recall calculation\n")
        precision = recall = None
    else:
        print("calculating precision recall...")
        precision, recall = precision_and_recall(generated_features, dataset_features)
        print(f"precision: {precision}")
        print(f"recall: {recall}\n")

    #plot histogram of predictions
    yhat = generated_predictions.max(dim=1).indices
    fig_hist_generated = plt.figure()
    plt.bar(*np.unique(yhat.cpu(), return_counts=True))
    plt.title(f'generated {args.ckpt}')
    try:
        plt.show()
    except:
        pass  # in some configurations the image cannot be shown

    yhat = dataset_predictions.max(dim=1).indices
    fig_hist_dataset = plt.figure()
    plt.bar(*np.unique(yhat.cpu(), return_counts=True))
    plt.title('dataset')
    try:
        plt.show()
    except:
        pass  # in some configurations the image cannot be shown

    if not TEST:
        save_results(args, fid, kid, (generated_diversity, dataset_diversity),
                 (precision, recall), fig_hist_generated, fig_hist_dataset)


def save_results(args, fid, kid, diversity, prec_rec, fig_hist_g, fig_hist_r):

    # define output path
    if args.out_path is not None:
        out_path = args.out_path
        os.makedirs(out_path, exist_ok=True)
    else:
        time_str = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
        out_path = osp.join(osp.splitext(args.ckpt)[0] + '_files', f'{time_str}_eval')
        os.makedirs(out_path, exist_ok=True)

    # same numeric results
    num_res_path = osp.join(out_path, 'eval.csv')
    pd.Series({'fid': fid, 'kid':kid, 'diversity':diversity, 'prec_rec':prec_rec}).to_csv(num_res_path, sep='\t', header=None) # save args
    args_path = osp.join(out_path, 'args.csv')
    pd.Series(args.__dict__).to_csv(args_path, sep='\t', header=None) # save args

    # save distribution images
    fig_name = osp.join(out_path, 'distribution_g.png')
    fig_hist_g.savefig(fig_name)
    fig_name = osp.join(out_path, 'distribution_r.png')
    fig_hist_r.savefig(fig_name)


if __name__ == '__main__':
    main(_sys.argv[1:])
