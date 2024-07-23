import os
import os.path as osp
import sys
import random

from torch import optim
from torch.utils import data
import torch.distributed as dist
from torch.utils.data import TensorDataset
from tqdm import tqdm

from models.encoder_mask import ConditionalMask, ReconLoss, PositionLoss, ContactLabelLoss, GlobalPosLoss
from models.inverse_losses import DiscriminatorLoss, LatentCenterRegularizer, FootContactUnsupervisedLoss
from utils.visualization import motion2bvh_rot
from utils.data import calc_bone_lengths, sample_data, requires_grad, data_sampler
from utils.traits import *
from models.gan import Discriminator
from motion_class import DynamicData

from utils.distributed import (
    get_rank,
    synchronize,
)

try:
    from clearml import Task
except ImportError:
    Task = None

try:
    from utils.loss_recorder import LossRecorder
except ImportError:
    LossRecorder = None

from utils.pre_run import TrainEncoderOptions, load_all_form_checkpoint


def train_enc(args, loader, encoder, e_optim, d_optim, g_ema, device, motion_statics, logger, make_mask,
              discriminator, latent_center, normalisation_data):
    loader = sample_data(loader)
    recon_criteria = ReconLoss(args.loss_type)
    pos_loss_local = PositionLoss(motion_statics, True, args.foot, args.use_velocity,
                                   normalisation_data['mean'], normalisation_data['std'], local_frame=args.use_local_pos)
    contact_criteria = ContactLabelLoss()
    global_pos_criteria = GlobalPosLoss(args)
    foot_contact_criteria = FootContactUnsupervisedLoss(motion_statics, normalisation_data, args.glob_pos, args.use_velocity)
    discriminator_criteria = DiscriminatorLoss(args, discriminator)
    latent_center_criteria = LatentCenterRegularizer(args, latent_center)
    pbar = range(args.iter)

    mean_path_length = 0

    if get_rank() == 0 and not args.on_cluster_training:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=False, smoothing=0.01, ncols=150)

    loss_dict = {}

    if args.distributed:
        g_module = g_ema.module
        e_module = encoder.module

    else:
        g_module = g_ema
        e_module = encoder

    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    requires_grad(g_ema, False)

    zero = torch.tensor(0.).to(device)

    from train import d_logistic_loss, get_grad_mean_max, d_r1_loss, g_nonsaturating_loss, g_path_regularize

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        if args.train_with_generated and random.randint(0, 1) == 0:
            noise = torch.randn(args.batch, args.latent, device=device)
            with torch.no_grad():
                real_img, _, _ = g_ema([noise], input_is_latent=False, return_latents=False)

        else:
            real_img = next(loader)[0]  # joints x coords x frames
            real_img = real_img.float() # loader produces doubles (64 bit), where network uses floats (32 bit)
            real_img = real_img.transpose(1,2) #  joints x coords x frames  ==>   coords x joints x frames
            real_img = real_img.to(device)
        # if args.foot:
        #     real_img = append_foot_contact(args, real_img, edge_rot_dict_general)

        ######################
        # step discriminator #
        ######################
        if args.train_disc and i % args.disc_freq == 0:

            disc_mask = make_mask(real_img, indicator_only=True) if args.partial_disc else 1.

            requires_grad(encoder, False)
            requires_grad(discriminator, True)

            fake_img = make_mask(real_img)
            _, rec_latent, _ = encoder(fake_img)
            rec_img, _, _ = g_ema([rec_latent], input_is_latent=True)

            real_img_aug = real_img

            fake_pred, rec_latent = discriminator(disc_mask * rec_img)
            real_pred, rec_real_latent = discriminator(disc_mask * real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()


            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

            d_regularize = (i // args.disc_freq) % args.d_reg_every == 0

            if d_regularize:
                real_img.requires_grad = True
                real_pred, _= discriminator(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)

                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

                d_optim.step()
                real_img.requires_grad_(False)

            loss_dict["r1"] = r1_loss

        ######################
        # step encoder #
        ######################

        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        fake_img = make_mask(real_img)
        _, rec_latent = encoder(fake_img)
        rec_img, _, _ = g_ema([rec_latent], input_is_latent=True)


        if args.train_disc and args.partial_disc:
            fake_pred, rec_latent = discriminator(rec_img * disc_mask)
        else:
            fake_pred, rec_latent = discriminator(rec_img)


        rec_img_full = rec_img
        real_img_full = real_img
        if args.partial_loss:
            partial_mask = make_mask(real_img, indicator_only=True)
            rec_img = rec_img * partial_mask
            real_img = real_img * partial_mask

        rec_loss = recon_criteria(rec_img, real_img)
        pos_loss = pos_loss_local(rec_img, real_img)
        contact_loss = contact_criteria(rec_img, real_img)
        global_pos_loss = global_pos_criteria(rec_img, real_img)
        foot_contact_loss = foot_contact_criteria(rec_img) if args.lambda_foot_contact > 0. else zero
        disc_loss = discriminator_criteria(rec_img) if args.lambda_disc > 0. else zero
        reg_loss = latent_center_criteria(rec_latent) if args.lambda_reg > 0. else zero

        total_loss = args.lambda_rec * rec_loss +\
                     args.lambda_pos * pos_loss +\
                     args.lambda_contact * contact_loss +\
                     args.lambda_global_pos * global_pos_loss +\
                     args.lambda_foot_contact * foot_contact_loss +\
                     args.lambda_disc * disc_loss +\
                     args.lambda_reg * reg_loss

        encoder.zero_grad()
        total_loss.backward(retain_graph=True)
        e_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize and args.train_disc:
            fake_img = make_mask(real_img)
            _, latents, _ = encoder(fake_img)
            fake_img_path, _, _ = g_ema([latents], input_is_latent=True)


            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_path, latents, mean_path_length
            )

            encoder.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            weighted_path_loss.backward()

            e_optim.step()

        loss_dict["rec_loss"] = rec_loss.item()
        loss_dict["pos_loss"] = pos_loss.item()
        loss_dict["total_loss"] = total_loss.item()
        loss_dict["contact_loss"] = contact_loss.item()
        loss_dict["global_pos_loss"] = global_pos_loss.item()
        loss_dict["disc_loss"] = disc_loss.item()
        loss_dict["reg_loss"] = reg_loss.item()
        loss_dict["foot_contact_loss"] = foot_contact_loss.item()

        e_loss_val = total_loss.item()

        if get_rank() == 0:
            description_str = f"e: {e_loss_val:.4f}; pos: {pos_loss.item():.4f}; rec: {rec_loss.item():.4f}; contact: {contact_loss.item():.4f}; global_pos: {global_pos_loss.item():.4f};"
            if isinstance(pbar, tqdm):
                pbar.set_description(description_str)
            elif i % 100 == 0:
                print(f'[{i}/{args.iter}]', description_str)

            if args.clearml or args.tensorboard:
                for loss_name in loss_dict.keys():
                    logger.report_scalar("Losses", loss_name, iteration=i, value=loss_dict[loss_name])

            if i == 0 or (i + 1) % args.report_every == 0:
                motion_all = DynamicData(rec_img.detach().cpu(), motion_statics, use_velocity=args.use_velocity)
                motion2bvh_rot(motion_all, osp.join(args.model_save_path, f"bvhs/{i + 1:05d}.bvh"))
            if i == 0 or (i+1) % args.report_every == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "args": args,
                    },
                    osp.join(args.model_save_path, f"checkpoint/{str(i).zfill(6)}.pt")
                )


def prepare_recorder(args):
    if args.clearml:
        output_folder = osp.expanduser('~/train_outputs')
        os.makedirs(output_folder, exist_ok=True)

        task = Task.init(project_name='stylegan2_motion_skeleton',
                         task_name=args.name,  # 'Jasper_all_5K_no_norm_mixing_0p9_conv3_fan_in_revw',
                         output_uri=output_folder)
        logger = task.get_logger()
        task_destination = task._get_output_destination_suffix()
        images_output_folder = osp.join(output_folder, task_destination, 'images')
        animations_output_folder = osp.join(output_folder, task_destination, 'animations')
    elif args.tensorboard:
        output_folder = os.path.join(args.model_save_path, 'tensorboard_outputs')
        os.makedirs(output_folder, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output_folder)
        logger = LossRecorder(writer)
        images_output_folder = osp.join(args.model_save_path, 'images')
        animations_output_folder = osp.join(args.model_save_path, 'animations')
    else:
        output_folder = osp.expanduser('~/tmp')
        logger = None
        images_output_folder = osp.join(output_folder, 'images')
        animations_output_folder = osp.join(output_folder, 'animations')

    if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)
    if not os.path.exists(animations_output_folder):
        os.makedirs(animations_output_folder)
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(osp.join(args.model_save_path, 'checkpoint'), exist_ok=True)
    os.makedirs(osp.join(args.model_save_path, 'bvhs'), exist_ok=True)
    return logger, images_output_folder, animations_output_folder


def main(args_not_parsed):
    parser = TrainEncoderOptions()
    args = parser.parse_args(args_not_parsed)
    device = args.device

    g_ema, discriminator, motion_data, mean_latent, motion_statics , normalisation_data, args = load_all_form_checkpoint(args.ckpt_existing, args, return_motion_data=True)
    if args.overfitting:
        motion_data = motion_data[:args.overfitting]

    traits_class = g_ema.traits_class
    if args.n_latent_predict > 1:
        args.n_latent_predict = g_ema.n_latent

    logger, images_output_folder, animations_output_folder = prepare_recorder(args)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_mlp = 8 # num linear layers in the generator's mapping network (z to W)

    args.start_iter = 0

    encoder = Discriminator(traits_class=traits_class, motion_statics =motion_statics ,
                            latent_dim=args.latent,
                            latent_rec_idx=int(args.encoder_latent_rec_idx), n_latent_predict=args.n_latent_predict,
                            ).to(device)

    e_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.d_lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.d_lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    ) if args.train_disc else None

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        encoder.load_state_dict(ckpt["e"])

        e_optim.load_state_dict(ckpt["e_optim"])

    gt_bone_lengths = calc_bone_lengths(motion_data) if args.entity == 'Joint' else None
    motion_all = DynamicData(torch.from_numpy(motion_data[0]).transpose(0, 1), motion_statics , use_velocity=args.use_velocity)

    motion_path = osp.join(animations_output_folder, 'real_motion.bvh')
    motion2bvh_rot(motion_all, motion_path)

    if args.clearml:
        logger.report_media(title='Animation', series='Ground Truth Motion', iteration=0, local_path=motion_path)

    motions_data_torch = torch.from_numpy(motion_data)
    dataset = TensorDataset(motions_data_torch)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=args.overfitting == 0,
    )

    make_mask = ConditionalMask(args, n_frames=args.n_frames, keep_loc=args.keep_loc, keep_rot=args.keep_rot,
                                normalisation_data=normalisation_data, noise_level=args.noise_level)
    
    normalisation_data = {'mean': torch.from_numpy(normalisation_data['mean']).to(device),
                          'std': torch.from_numpy(normalisation_data['std']).to(device)}
    
    train_enc(args, loader, encoder, e_optim, d_optim, g_ema, device, motion_statics , logger, make_mask,
              discriminator, mean_latent, normalisation_data)


if __name__ == "__main__":
    main(sys.argv[1:])
