import os
import os.path as osp
import sys
import datetime
import pandas as pd

from models.encoder_mask import ConditionalMask, ReconLoss, PositionLoss, PositionLossRoot
import functools
import pickle

from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from utils.visualization import motion2bvh
from utils.traits import *
from utils.data import to_list_4D, un_normalize

from models.gan import Generator, Discriminator
from utils.data import Joint, Edge, requires_grad, data_sampler  # used by eval
from evaluate import initialize_model
from utils.pre_run import TestEncoderOptions, load_all_form_checkpoint
from utils.interactive_utils import blend

from Motion.Quaternions import Quaternions

def inject_with_latent(rec_latent, g_ema, inject_idx=6):
    noise = torch.randn_like(rec_latent)
    latent = g_ema.get_latent(noise)
    # latents = [[rec_latent[i], latent[i]] for i in range(img.shape[0])]
    latents = torch.cat((rec_latent[:, :inject_idx], latent[:, inject_idx:]), dim=1)
    return latents


def eval_input(img, encoder, g_ema, device, mask_make=None, n_frames=None, mixed_generation=False):
    if mask_make is None:
        mask_make = lambda x, **kargs: x

    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = img.float().to(device)
    img = img.transpose(1, 2)
    img = mask_make(img, cond_length=n_frames)
    _, rec_latent, _ = encoder(img)

    if mixed_generation:
        latents = [inject_with_latent(rec_latent, g_ema, inject_idx=6)]
    else:
        latents = [rec_latent]

    rec_img, _, _ = g_ema(latents, input_is_latent=True)

    if img.shape[1] == 5:
        img = img[:, :4]
    return rec_img, img


def test_encoder(args, loader, encoder, g_ema, device, edge_rot_dict_general, make_mask,
                 mean_latent, discriminator=None):
    l1_loss = ReconLoss('L1')
    l2_loss = ReconLoss('L2')
    pos_loss = PositionLoss(edge_rot_dict_general, device, True, args.foot)
    pos_loss_local = PositionLoss(edge_rot_dict_general, device, True, args.foot, local_frame=True)
    pos_loss_root = PositionLossRoot(edge_rot_dict_general, device, True, args.foot)
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
        _, rec_latent, _ = encoder(fake_img)
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

    if not 'mask_extra_channel' in args:
        args.mask_extra_channel = 0
    if not 'mask_fill_noise' in args:
        args.mask_fill_noise = 0
    if not 'variable_mask' in args:
        args.variable_mask = 0

    # override arguments that were not used when training encoder
    if test_args.ckpt_existing is not None:
        args.ckpt_existing = test_args.ckpt_existing

    if not hasattr(args, 'num_convs'):
        args.num_convs = 1

    from utils.pre_run import setup_env
    traits_class = setup_env(args, True)

    args.n_mlp = 8 # num linear layers in the generator's mapping network (z to W)

    entity = eval(args.entity)

    g_ema, discriminator, motion_data, mean_latent, edge_rot_dict_general = load_all_form_checkpoint(args.ckpt_existing, test_args, return_motion_data=True, return_edge_rot_dict_general=True)

    encoder = Discriminator(traits_class=traits_class, entity=entity,
                            latent_dim=args.latent,
                            latent_rec_idx=int(args.encoder_latent_rec_idx), n_latent_predict=args.n_latent_predict,
                            mask_extra_channel=args.mask_extra_channel
                            ).to(device)
    encoder.load_state_dict(ckpt['e'])

    eval_ids = test_args.eval_id

    make_mask = ConditionalMask(args, n_frames=args.n_frames,
                                keep_loc=args.keep_loc, keep_rot=args.keep_rot,
                                edge_rot_dict_general=edge_rot_dict_general,
                                variable_mask=args.variable_mask)

    type2func = {'inversion': inversion, 'fusion': motion_fusion, 'editing': manual_editing ,
                 'editing_seed': manual_editing_seed, 'denoising': denoising, 'auto_regressive': auto_regressive }
    type2func[test_args.application](args, device, edge_rot_dict_general, encoder, eval_ids, g_ema, motion_data,
                                     output_path_anim, test_args, make_mask)
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

        loss_dict = test_encoder(args, loader, encoder, g_ema, device, edge_rot_dict_general, make_mask,
                                 mean_latent, discriminator)
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
        img = img.transpose(1, 2)

        make_mask.n_frames = 32

        n_cond_frames = make_mask.n_frames
        final_res = img[..., :n_cond_frames]
        while final_res.shape[-1] < n_frames:
            new_input = torch.empty_like(img)
            new_input[..., :n_cond_frames] = final_res[..., -n_cond_frames:]
            new_input = make_mask(new_input, cond_length=make_mask.n_frames)

            _, rec_latent, _ = encoder(new_input)
            rec_latent = inject_with_latent(rec_latent, g_ema, inject_idx=6)
            rec_img, _, _ = g_ema([rec_latent], input_is_latent=True, truncation=1, truncation_latent=g_ema.mean_latent)

            if discard_default:
                rec_img = rec_img[..., :-discard_default]
            final_res = blend(final_res, rec_img, final_res.shape[-1], 16, n_cond_frames)

        final_res = final_res[..., :n_frames]
        return final_res


def auto_regressive(args, device, edge_rot_dict_general, encoder, eval_ids, g_ema, motion_data, output_path_anim, test_args, make_mask):
    output_path_anim_id = osp.join(output_path_anim, 'auto_regressive')
    os.makedirs(output_path_anim_id, exist_ok=True)
    save_bvh = functools.partial(motion2bvh, edge_rot_dict_general=edge_rot_dict_general, entity='Edge')
    noise = torch.randn(args.batch, args.latent, device=device)
    with torch.no_grad():
        start_img, _, _ = g_ema([noise], input_is_latent=False, return_latents=False)
        start_img = start_img.permute(0, 2, 1, 3)
        pass
    res = auto_regressive_exec(start_img, encoder, g_ema, device, make_mask, test_args.n_frames_autoregressive)
    save_bvh(res.permute(0, 2, 1, 3).detach().cpu().numpy(), osp.join(output_path_anim_id, 'auto_regressive.bvh'))
    save_bvh(start_img.detach().cpu().numpy(), osp.join(output_path_anim_id, 'auto_regressive_start.bvh'))



def inversion(args, device, edge_rot_dict_general, encoder, eval_ids, g_ema, motion_data, output_path_anim, test_args, make_mask):
    output_path_anim_id = osp.join(output_path_anim, 'inversion')
    os.makedirs(output_path_anim_id, exist_ok=True)
    for eval_id in tqdm(eval_ids):
        motion_data_eval = torch.from_numpy(motion_data[[eval_id]]).float().to(device)

        save_bvh = functools.partial(motion2bvh, edge_rot_dict_general=edge_rot_dict_general, entity='Edge')

        output = eval_input(motion_data_eval, encoder, g_ema, device, make_mask)[0]
        save_bvh(output.permute(0, 2, 1, 3).detach().cpu().numpy(),
                 osp.join(output_path_anim_id, '{}_{}.bvh').format(eval_id, 'inversion'))
        save_bvh(motion_data_eval.detach().cpu().numpy(), osp.join(output_path_anim_id, '{}_gt.bvh').format(eval_id))


def motion_fusion(args, device, edge_rot_dict_general, encoder, eval_ids, g_ema, motion_data, output_path_anim,
                   test_args, make_mask):
    output_path_anim_id = osp.join(output_path_anim, f'motion_fusion')
    os.makedirs(output_path_anim_id, exist_ok=True)
    frame_idx = 40
    motion_data_first = torch.from_numpy(motion_data[[eval_ids[0]]]).float().to(device)
    motion_data_second = torch.from_numpy(motion_data[[eval_ids[1]]]).float().to(device)
    # create mixed motion
    motion_data_eval = motion_data_first.clone()
    motion_data_eval[..., frame_idx:] = motion_data_second[...,frame_idx:]

    save_bvh = functools.partial(motion2bvh, edge_rot_dict_general=edge_rot_dict_general, entity='Edge')
    output = eval_input(motion_data_eval, encoder, g_ema, device, make_mask)[0]

    save_bvh(output.permute(0, 2, 1, 3).detach().cpu().numpy(),
             osp.join(output_path_anim_id, '{}_{}.bvh').format(f'{eval_ids[0]}_{eval_ids[1]}', 'fusion'))
    save_bvh(motion_data_eval.detach().cpu().numpy(),
             osp.join(output_path_anim_id, '{}_transition_gt.bvh').format(f'{eval_ids[0]}_{eval_ids[1]}'))


def manual_editing(args, device, edge_rot_dict_general, encoder, eval_ids, g_ema, motion_data, output_path_anim,
                   test_args, make_mask):
    output_path_anim_id = osp.join(output_path_anim, f'spatial_editing')
    os.makedirs(output_path_anim_id, exist_ok=True)
    for eval_id in tqdm(eval_ids):
        motion_data_eval = torch.from_numpy(motion_data[[eval_id]]).float().to(device)
        save_bvh = functools.partial(motion2bvh, edge_rot_dict_general=edge_rot_dict_general, entity='Edge')
        joint_idx = 12
        order ='xyz'
        for axis in [1]:
            motion_data_edit = to_list_4D(motion_data_eval.detach().cpu().numpy())
            motion_data_edit = un_normalize(motion_data_edit, mean=edge_rot_dict_general['mean'].transpose(0, 2, 1, 3),
                                       std=edge_rot_dict_general['std'].transpose(0, 2, 1, 3))

            motion_data_edit = motion_data_edit[0]
            for idx in range(motion_data_edit[0,joint_idx,:,:].shape[1]):
                if idx>32:
                    quat_abs = abs(Quaternions(motion_data_edit[0,joint_idx,:,idx]))
                    angle_rad_xyz = quat_abs.euler(order=order)
                    angle_deg = np.rad2deg(angle_rad_xyz)
                    angle_deg[0][axis] += 110
                    angle_rad = np.deg2rad(angle_deg)
                    n_quat = torch.tensor(np.asarray(Quaternions.from_euler(angle_rad, order=order, world=True)))
                    motion_data_edit[0,joint_idx,:,:][:,idx] = n_quat

            # normalize again
            mean = edge_rot_dict_general['mean'].transpose(0, 2, 1, 3)
            std = edge_rot_dict_general['std'].transpose(0, 2, 1, 3)
            motion_data_edit = (motion_data_edit - mean) / std

            output = eval_input(motion_data_edit, encoder, g_ema, device, make_mask)[0]
            save_bvh(output.permute(0, 2, 1, 3).detach().cpu().numpy(),
                     osp.join(output_path_anim_id, '{}_{}.bvh').format(f'{eval_id}', 'result'))
            save_bvh(motion_data_edit,
                     osp.join(output_path_anim_id, '{}_gt_edited.bvh').format(eval_id))
            save_bvh(motion_data_eval.detach().cpu().numpy(),
                     osp.join(output_path_anim_id, '{}_gt.bvh').format(eval_id))


def manual_editing_seed(args, device, edge_rot_dict_general, encoder, seeds, g_ema, motion_data, output_path_anim,
                   test_args, make_mask):
    output_path_anim_id = osp.join(output_path_anim, f'spatial_editing_seed')
    os.makedirs(output_path_anim_id, exist_ok=True)
    joint_idx = 12
    frame_to_edit_from = 32
    for seed in tqdm(seeds):
        rnd_generator = torch.Generator(device=device).manual_seed(int(seed))
        sample_z = torch.randn(1, args.latent, device=device, generator=rnd_generator)
        motion_data_eval, W, _ = g_ema(
            [sample_z], truncation_latent=g_ema.mean_latent(args.truncation_mean),
            return_sub_motions=False, return_latents=True)
        motion_data_eval = motion_data_eval.transpose(2,1)

        save_bvh = functools.partial(motion2bvh, edge_rot_dict_general=edge_rot_dict_general, entity='Edge')
        order = 'xyz'
        for axis in [1]:
            motion_data_edit = to_list_4D(motion_data_eval.detach().cpu().numpy())
            motion_data_edit = un_normalize(motion_data_edit, mean=edge_rot_dict_general['mean'].transpose(0, 2, 1, 3),
                                        std=edge_rot_dict_general['std'].transpose(0, 2, 1, 3))
            motion_data_edit = motion_data_edit[0]
            for idx in range(motion_data_edit[0,joint_idx,:,:].shape[1]):
                if idx>frame_to_edit_from:
                    quat_abs = abs(Quaternions(motion_data_edit[0,joint_idx,:,idx]))
                    # quat_abs = Quaternions(motion_data_eval[0,joint_idx,:,:][:,idx].detach().cpu().numpy())
                    angle_rad_xyz = quat_abs.euler(order=order)
                    angle_deg = np.rad2deg(angle_rad_xyz)
                    angle_deg[0][axis] += 120
                    # angle_deg[0][2] += 60 #add this for jump motion and frame=20

                    angle_rad = np.deg2rad(angle_deg)
                    n_quat = torch.tensor(np.asarray(Quaternions.from_euler(angle_rad, order=order, world=True)))
                    motion_data_edit[0,joint_idx,:,:][:,idx] = n_quat

            mean = edge_rot_dict_general['mean'].transpose(0, 2, 1, 3)
            std = edge_rot_dict_general['std'].transpose(0, 2, 1, 3)
            motion_data_edit = (motion_data_edit - mean) / std

        output = eval_input(motion_data_edit, encoder, g_ema, device, make_mask)[0]
        save_bvh(output.permute(0, 2, 1, 3).detach().cpu().numpy(),
                 osp.join(output_path_anim_id, '{}_{}.bvh').format(f'{seed}', 'result'))
        save_bvh(motion_data_edit,
                 osp.join(output_path_anim_id, '{}_gt_edited.bvh').format(seed))
        save_bvh(motion_data_eval.detach().cpu().numpy(),
                 osp.join(output_path_anim_id, '{}_gt.bvh').format(seed))


def denoising(args, device, edge_rot_dict_general, encoder, eval_ids, g_ema, motion_data, output_path_anim,
                   test_args, make_mask):
    output_path_anim_id = osp.join(output_path_anim, f'denoising')
    np.random.seed(0)
    torch.manual_seed(0)

    for eval_id in tqdm(eval_ids):
        motion_data_eval = torch.from_numpy(motion_data[[eval_id]]).float().to(device)
        save_bvh = functools.partial(motion2bvh, edge_rot_dict_general=edge_rot_dict_general, entity='Edge')

        noise_mean = 0
        noise_std = 0.1

        save_bvh(motion_data_eval.detach().cpu().numpy(),
                 osp.join(output_path_anim_id, f'{eval_id:04d}_gt.bvh'))
        motion_data_eval += torch.randn(motion_data_eval.size()).cuda() * noise_std + noise_mean

        output = eval_input(motion_data_eval, encoder, g_ema, device, make_mask)[0]
        save_bvh(output.permute(0, 2, 1, 3).detach().cpu().numpy(),
                 osp.join(output_path_anim_id, f'{eval_id:04d}_denoise.bvh'))
        save_bvh(motion_data_eval.detach().cpu().numpy(),
                 osp.join(output_path_anim_id, f'{eval_id:04d}_noisy.bvh'))
        print("saved to ", output_path_anim_id)


if __name__ == "__main__":
    main(sys.argv[1:])
