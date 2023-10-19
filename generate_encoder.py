import os
import os.path as osp
import sys
import datetime
import pandas as pd
import torch

from models.encoder_mask import ConditionalMask, ReconLoss, PositionLoss, PositionLossRoot
import functools
import pickle

from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from utils.visualization import motion2bvh_rot
from utils.traits import *

from models.gan import Generator, Discriminator
from utils.data import requires_grad, data_sampler
from evaluate import initialize_model
from utils.pre_run import TestEncoderOptions, load_all_form_checkpoint
from utils.interactive_utils import blend

from Motion.Quaternions import Quaternions
from motion_class import StaticData, DynamicData

def inject_with_latent(rec_latent, g_ema, inject_idx=6):
    noise = torch.randn_like(rec_latent)
    latent = g_ema.get_latent(noise)
    latents = torch.cat((rec_latent[:, :inject_idx], latent[:, inject_idx:]), dim=1)
    return latents


def eval_input(img, encoder, g_ema, device, mask_make=None, n_frames=None, mixed_generation=False):
    if mask_make is None:
        mask_make = lambda x, **kargs: x

    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = img.float().to(device)
    img = mask_make(img, cond_length=n_frames)
    _, rec_latent = encoder(img)

    if mixed_generation:
        latents = [inject_with_latent(rec_latent, g_ema, inject_idx=6)]
    else:
        latents = [rec_latent]

    rec_img, _, _ = g_ema(latents, input_is_latent=True)

    if img.shape[1] == 5:
        img = img[:, :4]
    return rec_img, img


def test_encoder(args, loader, encoder, g_ema, device, motion_statics : StaticData, make_mask,
                 mean_latent, mean_joints, std_joints, discriminator=None):
    l1_loss = ReconLoss('L1')
    l2_loss = ReconLoss('L2')

    pos_loss = PositionLoss(motion_statics , True, args.foot, args.use_velocity, mean_joints, std_joints)
    pos_loss_local = PositionLoss(motion_statics , True, args.foot,
                                  args.use_velocity, mean_joints, std_joints, local_frame=True)
    pos_loss_root = PositionLossRoot(motion_statics , device, True, args.foot, args.use_velocity, mean_joints, std_joints)
    stgcn_model = initialize_model(device, args.action_recog_model, args.dataset)

    pbar = tqdm(enumerate(loader), initial=args.start_iter, dynamic_ncols=False, smoothing=0.01, ncols=150, total=len(loader))

    losses = {'l1_loss': [], 'l2_loss': [], 'pos_loss': [], 'pos_loss_local': [], 'pos_loss_root': []}

    requires_grad(g_ema, False)
    requires_grad(encoder, False)

    for idx, real_img in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = real_img[0]  # joints x coords x frames
        real_img = real_img.float().to(device) # loader produces doubles (64 bit), where network uses floats (32 bit)
        real_img = real_img.transpose(1, 2) #  joints x coords x frames  ==>   coords x joints x frames

        fake_img = make_mask(real_img)
        _, rec_latent = encoder(fake_img)
        rec_img, _, _ = g_ema([rec_latent], input_is_latent=True)
        for key in losses.keys():
            loss_criteria = eval(key)
            loss = loss_criteria(rec_img, real_img)
            losses[key].append(loss.item())

    keys = list(losses.keys())
    for key in keys:
        losses[key] = np.mean(losses[key])

    return losses


def main(args_not_parsed):
    parser = TestEncoderOptions()
    test_args = parser.parse_args(args_not_parsed)
    device = test_args.device

    if test_args.out_path is not None:
        output_path = test_args.out_path
    else:
        output_path = '/'.join(test_args.ckpt.split('/')[:-1])
        folder_name = f"{test_args.ckpt.split('/')[-1][:-3]}_files"
        output_path = osp.join(output_path, folder_name)
    os.makedirs(output_path,  exist_ok=True)
    output_path_anim = osp.join(output_path, 'motions')
    os.makedirs(output_path_anim, exist_ok=True)

    ckpt = torch.load(test_args.ckpt, map_location=device)
    args = ckpt['args']

    # override arguments that were not used when training encoder
    if test_args.ckpt_existing is not None:
        args.ckpt_existing = test_args.ckpt_existing

    if not hasattr(args, 'num_convs'):
        args.num_convs = 1

    from utils.pre_run import setup_env
    traits_class = setup_env(args, True)

    args.n_mlp = 8 # num linear layers in the generator's mapping network (z to W)

    g_ema, discriminator, motion_data, mean_latent, motion_statics , normalisation_data, args = load_all_form_checkpoint(args.ckpt_existing, test_args, return_motion_data=True)
    
    encoder = Discriminator(traits_class=traits_class, motion_statics =motion_statics ,
                            latent_dim=args.latent,
                            latent_rec_idx=int(args.encoder_latent_rec_idx), n_latent_predict=args.n_latent_predict,
                            ).to(device)
    encoder.load_state_dict(ckpt['e'])

    eval_ids = test_args.eval_id

    make_mask = ConditionalMask(args, n_frames=args.n_frames,
                                keep_loc=args.keep_loc, keep_rot=args.keep_rot,
                                normalisation_data=normalisation_data)

    mean_joints = torch.from_numpy(normalisation_data['mean']).to(device)
    std_joints = torch.from_numpy(normalisation_data['std']).to(device)

    type2func = {'inversion': inversion, 'fusion': motion_fusion, 'editing': manual_editing,
                 'editing_seed': manual_editing_seed, 'denoising': denoising, 'auto_regressive': auto_regressive}

    type2func[test_args.application](encoder, motion_statics , mean_joints, std_joints, g_ema, output_path_anim, make_mask,
                                     eval_ids=eval_ids, motion_data=motion_data.transpose(0, 2, 1, 3), use_velocity=args.use_velocity,
                                     device=device, args=args, test_args=test_args)
    if test_args.full_eval:
        np.random.seed(0)
        np.random.shuffle(motion_data)
        motion_data = motion_data[:100]
        motions_data_torch = torch.from_numpy(motion_data)

        dataset = TensorDataset(motions_data_torch)
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=False, distributed=args.distributed),
            drop_last=True,
        )

        loss_dict = test_encoder(args, loader, encoder, g_ema, device, motion_statics , make_mask,
                                 mean_latent, discriminator, mean_joints, std_joints)
        time_str = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
        output_path_losses = osp.join(output_path, f'losses')
        os.makedirs(output_path_losses, exist_ok=True)
        output_path_pickle = osp.join(output_path, f'losses_pickle')
        os.makedirs(output_path_pickle, exist_ok=True)

        with open(osp.join(output_path_pickle, f'{test_args.model_name}_summary_data_{time_str}.pickle'), 'wb') as handle:
            pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        args_path = osp.join(output_path_losses, f'{test_args.model_name}.csv')
        pd.Series(loss_dict).to_csv(args_path, sep='\t', header=None)  # save args
        print(loss_dict)
        print()


def auto_regressive_exec(img, encoder, g_ema, device, make_mask, n_frames):
    with torch.no_grad():
        discard_default = 5
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = img.float().to(device)

        make_mask.n_frames = 32

        n_cond_frames = make_mask.n_frames
        final_res = img[..., :n_cond_frames]
        while final_res.shape[-1] < n_frames:
            new_input = torch.empty_like(img)
            new_input[..., :n_cond_frames] = final_res[..., -n_cond_frames:]
            new_input = make_mask(new_input, cond_length=make_mask.n_frames)

            _, rec_latent = encoder(new_input)
            rec_latent = inject_with_latent(rec_latent, g_ema, inject_idx=6)
            rec_img, _, _ = g_ema([rec_latent], input_is_latent=True, truncation=1, truncation_latent=g_ema.mean_latent)

            if discard_default:
                rec_img = rec_img[..., :-discard_default]
            final_res = blend(final_res, rec_img, final_res.shape[-1], 16, n_cond_frames)

        final_res = final_res[..., :n_frames]
        return final_res


def auto_regressive(encoder, motion_statics : StaticData, mean_joints: torch.Tensor, std_joints: torch.Tensor,
                    g_ema, output_path_anim, make_mask, use_velocity, device, args, test_args, **kwargs):
    output_path_anim_id = osp.join(output_path_anim, 'auto_regressive')
    os.makedirs(output_path_anim_id, exist_ok=True)

    noise = torch.randn(args.batch, args.latent, device=device)
    with torch.no_grad():
        start_img, _, _ = g_ema([noise], input_is_latent=False, return_latents=False)
        pass
    res = auto_regressive_exec(start_img, encoder, g_ema, device, make_mask, test_args.n_frames_autoregressive)

    out_motions_all = DynamicData(res.detach(), motion_statics , use_velocity=use_velocity)
    out_motions_all = out_motions_all.un_normalise(mean_joints, std_joints)

    input_motions_all = DynamicData(start_img, motion_statics , use_velocity=use_velocity)
    input_motions_all = input_motions_all.un_normalise(mean_joints, std_joints)

    motion2bvh_rot(out_motions_all, osp.join(output_path_anim_id, 'auto_regressive.bvh'))
    motion2bvh_rot(input_motions_all, osp.join(output_path_anim_id, 'auto_regressive_start.bvh'))


def inversion(encoder, motion_statics : StaticData, mean_joints: torch.Tensor, std_joints: torch.Tensor,
              g_ema, output_path_anim, make_mask, eval_ids, motion_data, use_velocity, device, **kwargs):
    output_path_anim_id = osp.join(output_path_anim, 'inversion')
    os.makedirs(output_path_anim_id, exist_ok=True)
    for eval_id in tqdm(eval_ids):
        motion_data_eval = torch.from_numpy(motion_data[[eval_id]]).float().to(device)

        output = eval_input(motion_data_eval, encoder, g_ema, device, make_mask)[0]

        out_motions_all = DynamicData(output.detach(), motion_statics , use_velocity=use_velocity)
        out_motions_all = out_motions_all.un_normalise(mean_joints, std_joints)

        input_motions_all = DynamicData(motion_data_eval, motion_statics , use_velocity=use_velocity)
        input_motions_all = input_motions_all.un_normalise(mean_joints, std_joints)

        motion2bvh_rot(out_motions_all, osp.join(output_path_anim_id, f'{eval_id}_inversion.bvh'))
        motion2bvh_rot(input_motions_all, osp.join(output_path_anim_id, f'{eval_id}_gt.bvh'))


def motion_fusion(encoder, motion_statics : StaticData, mean_joints: torch.Tensor, std_joints: torch.Tensor,
              g_ema, output_path_anim, make_mask, eval_ids, motion_data, use_velocity, device, **kwargs):
    output_path_anim_id = osp.join(output_path_anim, f'motion_fusion')
    os.makedirs(output_path_anim_id, exist_ok=True)
    frame_idx = 40
    motion_data_first = torch.from_numpy(motion_data[[eval_ids[0]]]).float().to(device)
    motion_data_second = torch.from_numpy(motion_data[[eval_ids[1]]]).float().to(device)
    # create mixed motion
    motion_data_eval = motion_data_first.clone()
    motion_data_eval[..., frame_idx:] = motion_data_second[...,frame_idx:]

    output = eval_input(motion_data_eval, encoder, g_ema, device, make_mask)[0]

    out_motions_all = DynamicData(output.detach(), motion_statics , use_velocity=use_velocity)
    out_motions_all = out_motions_all.un_normalise(mean_joints, std_joints)

    input_motions_all = DynamicData(motion_data_eval, motion_statics , use_velocity=use_velocity)
    input_motions_all = input_motions_all.un_normalise(mean_joints, std_joints)

    motion2bvh_rot(out_motions_all, osp.join(output_path_anim_id, f'{eval_ids[0]}_{eval_ids[1]}_fusion.bvh'))
    motion2bvh_rot(input_motions_all, osp.join(output_path_anim_id, f'{eval_ids[0]}_{eval_ids[1]}_transition_gt.bvh'))


def manual_editing(encoder, motion_statics : StaticData, mean_joints: torch.Tensor, std_joints: torch.Tensor,
                    g_ema, output_path_anim, make_mask, eval_ids, motion_data, use_velocity, device, **kwargs):
    output_path_anim_id = osp.join(output_path_anim, f'spatial_editing')
    os.makedirs(output_path_anim_id, exist_ok=True)
    for eval_id in tqdm(eval_ids):
        motion_data_eval = torch.from_numpy(motion_data[[eval_id]]).float().to(device)
        joint_idx = 12
        order ='xyz'
        for axis in [1]:
            motion_data_edit = (motion_data_eval * std_joints + mean_joints)

            for idx in range(motion_data_edit[0,:, joint_idx,:].shape[1]):
                if idx>32:
                    quat_abs = abs(Quaternions(motion_data_edit[0,:, joint_idx,idx].cpu().numpy()))
                    angle_rad_xyz = quat_abs.euler(order=order)
                    angle_deg = np.rad2deg(angle_rad_xyz)
                    angle_deg[0][axis] += 110
                    angle_rad = np.deg2rad(angle_deg)
                    n_quat = torch.tensor(np.asarray(Quaternions.from_euler(angle_rad, order=order, world=True)))
                    motion_data_edit[0,:, joint_idx,:][:,idx] = n_quat

            # normalize again
            motion_data_edit = (motion_data_edit - mean_joints) / std_joints

            output = eval_input(motion_data_edit, encoder, g_ema, device, make_mask)[0]

            out_motions_all = DynamicData(output.detach(), motion_statics , use_velocity=use_velocity)
            out_motions_all = out_motions_all.un_normalise(mean_joints, std_joints)

            edit_motions_all = DynamicData(motion_data_edit, motion_statics , use_velocity=use_velocity)
            edit_motions_all = edit_motions_all.un_normalise(mean_joints, std_joints)

            input_motions_all = DynamicData(motion_data_eval, motion_statics , use_velocity=use_velocity)
            input_motions_all = input_motions_all.un_normalise(mean_joints, std_joints)

            motion2bvh_rot(out_motions_all, osp.join(output_path_anim_id, f'{eval_id}_result.bvh'))
            motion2bvh_rot(edit_motions_all, osp.join(output_path_anim_id, f'{eval_id}__gt_edited.bvh'))
            motion2bvh_rot(input_motions_all, osp.join(output_path_anim_id, f'{eval_id}__gt.bvh'))


def manual_editing_seed(encoder, motion_statics : StaticData, mean_joints: torch.Tensor, std_joints: torch.Tensor,
                        g_ema, output_path_anim, make_mask, eval_ids, use_velocity, device, args, **kwargs):
    output_path_anim_id = osp.join(output_path_anim, f'spatial_editing_seed')
    os.makedirs(output_path_anim_id, exist_ok=True)
    joint_idx = 12
    frame_to_edit_from = 32
    for seed in tqdm(eval_ids):
        rnd_generator = torch.Generator(device=device).manual_seed(int(seed))
        sample_z = torch.randn(1, args.latent, device=device, generator=rnd_generator)
        motion_data_eval, W, _ = g_ema(
            [sample_z], truncation_latent=g_ema.mean_latent(args.truncation_mean),
            return_sub_motions=False, return_latents=True)

        order = 'xyz'
        for axis in [1]:
            motion_data_edit = (motion_data_eval * std_joints + mean_joints)

            for idx in range(motion_data_edit[0,:, joint_idx,:].shape[1]):
                if idx>frame_to_edit_from:
                    quat_abs = abs(Quaternions(motion_data_edit[0,:, joint_idx,idx].detach().cpu().numpy()))
                    angle_rad_xyz = quat_abs.euler(order=order)
                    angle_deg = np.rad2deg(angle_rad_xyz)
                    angle_deg[0][axis] += 120
                    # angle_deg[0][2] += 60 #add this for jump motion and frame=20

                    angle_rad = np.deg2rad(angle_deg)
                    n_quat = torch.tensor(np.asarray(Quaternions.from_euler(angle_rad, order=order, world=True)))
                    motion_data_edit[0,:, joint_idx,:][:,idx] = n_quat

            motion_data_edit = (motion_data_edit - mean_joints) / std_joints
        
        output = eval_input(motion_data_edit, encoder, g_ema, device, make_mask)[0]

        out_motions_all = DynamicData(output, motion_statics , use_velocity=use_velocity)
        out_motions_all = out_motions_all.un_normalise(mean_joints, std_joints)

        edit_motions_all = DynamicData(motion_data_edit, motion_statics , use_velocity=use_velocity)
        edit_motions_all = edit_motions_all.un_normalise(mean_joints, std_joints)

        input_motions_all = DynamicData(motion_data_eval, motion_statics , use_velocity=use_velocity)
        input_motions_all = input_motions_all.un_normalise(mean_joints, std_joints)

        motion2bvh_rot(out_motions_all, osp.join(output_path_anim_id, f'{seed}_result.bvh'))
        motion2bvh_rot(edit_motions_all, osp.join(output_path_anim_id, f'{seed}__gt_edited.bvh'))
        motion2bvh_rot(input_motions_all, osp.join(output_path_anim_id, f'{seed}__gt.bvh'))


def denoising(encoder, motion_statics : StaticData, mean_joints: torch.Tensor, std_joints: torch.Tensor,
              g_ema, output_path_anim, make_mask, eval_ids, motion_data, use_velocity, device, **kwargs):
    output_path_anim_id = osp.join(output_path_anim, f'denoising')
    np.random.seed(0)
    torch.manual_seed(0)

    for eval_id in tqdm(eval_ids):
        motion_data_eval = torch.from_numpy(motion_data[[eval_id]]).float().to(device)

        noise_mean = 0
        noise_std = 0.1

        noise = torch.randn(motion_data_eval.size()).cuda() * noise_std + noise_mean
        motion_data_noisy = motion_data_eval.clone() + noise

        output = eval_input(motion_data_noisy, encoder, g_ema, device, make_mask)[0]

        out_motions_all = DynamicData(output.detach(), motion_statics , use_velocity=use_velocity)
        out_motions_all = out_motions_all.un_normalise(mean_joints, std_joints)

        input_motions_all = DynamicData(motion_data_eval, motion_statics, use_velocity=use_velocity)
        input_motions_all = input_motions_all.un_normalise(mean_joints, std_joints)

        noisy_motions_all = DynamicData(motion_data_eval, motion_statics, use_velocity=use_velocity)
        noisy_motions_all = noisy_motions_all.un_normalise(mean_joints, std_joints)

        motion2bvh_rot(out_motions_all, osp.join(output_path_anim_id, f'{eval_id:04d}_denoise.bvh'))
        motion2bvh_rot(input_motions_all, osp.join(output_path_anim_id, f'{eval_id:04d}_gt.bvh'))
        motion2bvh_rot(noisy_motions_all, osp.join(output_path_anim_id, f'{eval_id:04d}_noisy.bvh'))


if __name__ == "__main__":
    main(sys.argv[1:])
