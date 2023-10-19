import os
import re
import datetime
import os.path as osp

import torch
import sys as _sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import to_cpu
from models.gan import Generator
from motion_class import DynamicData, StaticData
from utils.visualization import motion2fig, motion_to_bvh
from utils.pre_run import GenerateOptions, load_all_form_checkpoint


def interpolate(args, g_ema, device, mean_latent):
    print('Interpolating...')
    num_interp = args.motions
    for interp_seeds in args.interp_seeds:
        assert len(interp_seeds) in range(1,3)
    generated_motions = []
    for interp_seeds in args.interp_seeds:
        seed_from = interp_seeds[0]
        seed_to = None if len(interp_seeds) < 2 else interp_seeds[1]

        rnd_generator = torch.Generator(device=device).manual_seed(seed_from)
        sample_z_from = torch.randn(1, args.latent, device=device, generator=rnd_generator)
        W_from = g_ema.get_latent(sample_z_from)

        if seed_to is not None:
            rnd_generator = torch.Generator(device=device).manual_seed(seed_to)
            sample_z_to = torch.randn(1, args.latent, device=device, generator=rnd_generator)
            W_to = g_ema.get_latent(sample_z_to)
        else:
            W_to = mean_latent

        generated_motion = [None] * num_interp
        steps = torch.linspace(0, 1, num_interp, device=device)

        for interp_idx in np.arange(num_interp):
            cur_W = W_from.lerp(W_to, steps[interp_idx])
            interpolated_motion, _, _ = g_ema(
                [cur_W],
                truncation=1,
                input_is_latent=True
            )
            generated_motion[interp_idx] = to_cpu(interpolated_motion)

        interp_name = 'interp_{}-{}'.format(interp_seeds[0],'mean' if len(interp_seeds)==1 else interp_seeds[-1])
        generated_motions.append((interp_name, generated_motion))

    return tuple(generated_motions)


def z_from_seed(args, seed, device):
    rnd_generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, args.latent, device=device, generator=rnd_generator)
    return z


def sample(args, g_ema, device, mean_latent):
    print('Sampling...')

    seed_rnd_mult = args.motions*10000
    if args.sample_seeds is None:
        seeds = np.array([])
        if args.no_idle:
            no_idle_thresh = 0.7  # hard coded for now. better compute the mean of all stds and set the threshold accordingly
            n_motions = 5*args.motions
            stds = np.zeros(n_motions)
        else:
            n_motions = args.motions
        while np.unique(seeds).shape[0] != n_motions: # refrain from duplicates in seeds
            seeds = (np.random.random(n_motions)*seed_rnd_mult).astype(int)
    else:
        seeds = np.array(args.sample_seeds)
    generated_motion = pd.DataFrame(index=seeds, columns=['motion', 'W'], dtype=object)
    for i, seed in enumerate(seeds):
        rnd_generator = torch.Generator(device=device).manual_seed(int(seed))
        sample_z = torch.randn(1, args.latent, device=device, generator=rnd_generator)
        motion, W, _ = g_ema(
            [sample_z], truncation=args.truncation, truncation_latent=mean_latent,
            return_sub_motions=args.return_sub_motions, return_latents=True)
        if args.no_idle:
            stds[i] = get_motion_std(args, motion)
        if (i+1) % 1000 == 0:
            print(f'Done sampling {i+1} motions.')

        # to_cpu is used becuase advanced python versions cannot assign a cuda object to a dataframe
        generated_motion.loc[seed,'motion'] = to_cpu(motion)
        generated_motion.loc[seed, 'W'] = to_cpu(W)

    if args.no_idle:
        filter = (stds > no_idle_thresh)
    else:
        filter = np.ones(generated_motion.shape[0], dtype=bool)
    generated_motion = generated_motion[filter]
    return generated_motion


def get_motion_std(args, motion):
    if args.entity == 'Edge' and args.glob_pos:
        assert args.foot
        std = motion[:, :3, -3, :].norm(p=2, dim=1).std()
    else:
        raise 'this case is not supported yet'
    return std


def edit(args, g_ema, device, mean_latent):
    boundary = np.load(args.boundary_path, allow_pickle=True)
    if isinstance(boundary[0], dict):
        boundary_normal = boundary[0]['normal']
    else: # backward compatibility to old format
        boundary_normal = boundary
    linspace = np.linspace(-args.edit_radius, args.edit_radius, 7)

    seeds = np.array(args.sample_seeds)
    generated = pd.DataFrame(index=seeds, columns=['motion', 'W', 'z'], dtype=object)

    for seed in seeds:
        generated.z[seed] = z_from_seed(args, seed, device)
    generated.W = generated.z.apply(g_ema.get_latent)

    interpolations = generated.W.apply(lambda W: W + torch.Tensor(linspace[:,np.newaxis] @ boundary_normal).to(device))
    generated.motion = interpolations.apply(lambda interp: g_ema([interp], truncation=1, input_is_latent=True))
    generated.motion = generated.motion.apply(lambda motion: motion[0])
    return generated.motion


def get_gen_mot_np(args, generated_motion, mean_joints, std_joints):
    # part 1: align data type
    if isinstance(generated_motion, pd.Series):
        index = generated_motion.index
        if not isinstance(generated_motion.iloc[0], list) and \
                generated_motion.iloc[0].ndim == 4 and generated_motion.iloc[0].shape[0] > 1:
            generated_motion = generated_motion.apply(
                lambda motions: torch.unsqueeze(motions, 1))  # add a batch dimension
            generated_motion = generated_motion.apply(list)   # part2 expects lists
        generated_motion = generated_motion.tolist()
    else:
        assert isinstance(generated_motion, list)
        index = range(len(generated_motion))

    # part 2: torch to np
    for i in np.arange(len(generated_motion)):
        if not isinstance(generated_motion[i], list):
            generated_motion[i] = generated_motion[i].transpose(1, 2).detach().cpu().numpy()
            assert generated_motion[i].shape[:3] == std_joints.shape[:3] or args.return_sub_motions
        else:
            generated_motion[i], _ = get_gen_mot_np(args, generated_motion[i], mean_joints, std_joints)

    return generated_motion, index


def generate(args, g_ema: Generator, device, mean_joints: torch.tensor, std_joints: torch.tensor,
             motion_statics: StaticData, entity: str):

    type2func = {'interp': interpolate, 'sample': sample, 'edit': edit}
    with torch.no_grad():
        g_ema.eval()
        mean_latent = g_ema.mean_latent(args.truncation_mean)
        generated_motions = type2func[args.type](args, g_ema, device, mean_latent)

    if args.out_path is not None:
        out_path = args.out_path
        os.makedirs(out_path, exist_ok=True)
    else:
        time_str = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
        out_path = osp.join(osp.splitext(args.ckpt)[0] + '_files', f'{time_str}_{args.type}')
        if args.type == 'sample':
            out_path = f'{out_path}_{args.motions}'
        os.makedirs(out_path, exist_ok=True)
    root_out_path = out_path

    generated_motions = (generated_motions,) if not isinstance(generated_motions, tuple) else generated_motions

    for generated_motion in generated_motions:
        out_path = root_out_path
        if isinstance(generated_motion, tuple):
            out_path = osp.join(out_path, generated_motion[0])
            os.makedirs(out_path, exist_ok=True)
            generated_motion = generated_motion[1]

        if not isinstance(generated_motion, pd.DataFrame):
            generated_motion = pd.DataFrame(columns=['motion'], data=generated_motion)

        # save W if exists
        if 'W' in generated_motion.columns:
            for seed in generated_motion.index:
                assert generated_motion.W[seed].ndim == 3 and generated_motion.W[seed].shape[0] == 1
                np.save(osp.join(out_path, 'Wplus_{}.npy'.format(seed)), generated_motion.W[seed][0].cpu().numpy())

        # save motions
        motion_np, _ = get_gen_mot_np(args, generated_motion['motion'], mean_joints, std_joints)
        prefix ='generated'

        # save one figure of several motions
        n_sampled_frames = 10
        n_motions = min(10, len(motion_np))

        motion_batch = torch.tensor([motion[0] for motion in motion_np]).permute(0, 2, 1, 3)
        motions_all = DynamicData(motion_batch, motion_statics, use_velocity=args.use_velocity)
        motions_all = motions_all.un_normalise(mean_joints.transpose(0, 2, 1, 3), std_joints.transpose(0, 2, 1, 3))
        fig = motion2fig(motions_all[:5], motion_statics.character_name, entity=entity)

        fig_name = osp.join(out_path, f'{prefix}.png')
        dpi = max(n_motions, n_sampled_frames) * 100
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        plt.close()

        for i, idx in enumerate(generated_motion.index):
            id = idx if idx is not None else i
            if args.simple_idx:
                id = '{:03d}'.format(i)
            if 'cluster_label' in generated_motion.columns:
                cluster_label = generated_motion.cluster_label[idx]
                cluster_label = torch.argmax(cluster_label).item()
                id = f'g{cluster_label:02d}_{id}'

            motion_to_bvh(motions_all[i], osp.join(out_path, f'{prefix}_{id}.bvh'), entity, motion_statics.parents_list[-1])

    # save args
    pd.Series(args.__dict__).to_csv(osp.join(root_out_path, 'args.csv'), sep='\t', header=None)
    print('saved to {}'.format(root_out_path))

    return root_out_path


def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def main(args_not_parsed):
    parser = GenerateOptions()
    args = parser.parse_args(args_not_parsed)
    device = args.device
    g_ema, _, _, motion_statics, mean_joints, std_joints, entity, args = load_all_form_checkpoint(args.ckpt, args)
    out_path = generate(args, g_ema, device, mean_joints, std_joints, motion_statics=motion_statics, entity=entity)
    return out_path


if __name__ == "__main__":
    main(_sys.argv[1:])
