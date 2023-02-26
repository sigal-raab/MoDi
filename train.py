import os
import os.path as osp
import time
import random
import math
import sys

from torch import autograd, optim
from torch.utils import data
import torch.distributed as dist
from torch.utils.data import TensorDataset
from tqdm import tqdm

from utils.visualization import motion2fig, motion2bvh
from utils.data import calc_bone_lengths
from utils.traits import *
from utils.data import foot_names
from utils.data import motion_from_raw

import matplotlib.pyplot as plt
import evaluate as evaluate

from models.gan import Generator, Discriminator
from utils.foot import get_foot_contact, get_foot_velo
from utils.data import Joint, Edge # to be used in 'eval'
from utils.pre_run import TrainOptions, setup_env

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
try:
    from clearml import Task
except ImportError:
    Task = None

try:
    from utils.loss_recorder import LossRecorder
except ImportError:
    LossRecorder = None

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        # refrain from computing gradients for parameters that are 'non_grad': masks, etc.
        if flag is False or \
                flag is True and hasattr(model, 'non_grad_params') and name not in model.non_grad_params:
            p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    # model1 <-- model1*decay + model2*(1-decay)
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

# discriminator pushes 'real' to be positive and 'fake' to be negative
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)  # softplus is a smooth ReLU
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


# generator pushes 'fake' to be positive
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()  # softplus is a smooth ReLU

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    # compute the gradient of an image, slightly perturbed, wrt the latents that created it
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def g_foot_contact_loss(motion, glob_pos, axis_up, edge_rot_dict_general):
    # motion is of shape samples x features x joints x frames
    label_idx = motion.shape[2] - len(foot_names)
    skeletal_foot_contact = get_foot_contact(motion[:, :, :label_idx], glob_pos, axis_up, edge_rot_dict_general, foot_names)
    predicted_foot_contact = motion[:, 0, label_idx:]
    return F.mse_loss(skeletal_foot_contact, predicted_foot_contact)
    # return F.binary_cross_entropy_with_logits((predicted_foot_contact-.5)*12, skeletal_foot_contact)


def sigmoid_for_contact(predicted_foot_contact):
    return torch.sigmoid((predicted_foot_contact - 0.5) * 2 * 6)


def g_foot_contact_loss_v2(motion, glob_pos, axis_up, edge_rot_dict_general):
    # motion is of shape samples x features x joints x frames
    label_idx = motion.shape[2] - len(foot_names)
    velo = get_foot_velo(motion[:, :, :label_idx], glob_pos, axis_up, edge_rot_dict_general)
    predicted_foot_contact = motion[:, 0, label_idx:]
    predicted_foot_contact = sigmoid_for_contact(predicted_foot_contact)
    return (predicted_foot_contact[..., 1:] * velo).mean()


def g_encourage_contact(motion):
    label_idx = motion.shape[2] - len(foot_names)
    predicted_foot_contact = motion[:, 0, label_idx:]
    predicted_foot_contact = sigmoid_for_contact(predicted_foot_contact)
    return F.binary_cross_entropy(predicted_foot_contact, torch.ones_like(predicted_foot_contact))


def average_contact_ratio(motion):
    label_idx = motion.shape[2] - len(foot_names)
    predicted_foot_contact = motion[:, 0, label_idx:]
    predicted_foot_contact = (predicted_foot_contact > 0.5).float()
    return predicted_foot_contact.mean(axis=(1, 2))


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    """ Genereate one or two noise arrays, depending on prob.
        one array: network will generate one W
        two arrays: network will generate two Ws and mix them """
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, logger, entity,
          animations_output_folder, images_output_folder, mean_joints=None, std_joints=None, gt_bone_lengths=None, edge_rot_dict_general=None):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0 and not args.on_cluster_training:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=False, smoothing=0.01, ncols=150)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    foot_contact_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    report_every = args.report_every
    time_measure = []
    start_time_measure = time.time()
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)[0]  # joints x coords x frames
        real_img = real_img.float() # loader produces doubles (64 bit), where network uses floats (32 bit)
        real_img = real_img.transpose(1,2) #  joints x coords x frames  ==>   coords x joints x frames
        real_img = real_img.to(device)

        ######################
        # step discriminator #
        ######################
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, gt_latents, inject_index = generator(noise, return_latents=True)

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        ##################
        # step generator #
        ##################
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, gt_latents, inject_index = generator(noise, return_latents=True,
                                                       return_sub_motions=args.return_sub_motions)
        fake_pred = discriminator(fake_img)

        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        # foot contact loss
        if args.foot:
            if args.v2_contact_loss:
                foot_contact_loss = g_foot_contact_loss_v2(fake_img, args.glob_pos, args.axis_up, edge_rot_dict_general)
            else:
                foot_contact_loss = g_foot_contact_loss(fake_img, args.glob_pos, args.axis_up, edge_rot_dict_general)
        loss_dict["foot_contact"] = foot_contact_loss

        loss_dict["encourage_contact"] = g_encourage_contact(fake_img)

        generator.zero_grad()
        (g_loss + args.g_foot_reg_weight * foot_contact_loss + args.g_encourage_contact_weight * loss_dict["encourage_contact"]).backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img_path, latents, _ = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_path, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img_path[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()  # handle distributed data
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        #  TODO: Replace with our evals
        # if i >= 2000 and i % 2000 == 0 and args.action_recog_model is not None:
        #     fid, kid, g_diversity = calc_evaluation_metrics(args, device, g_ema, entity, std_joints, mean_joints)
        #     loss_dict['evaluation_metrics_fid'] = fid
        #     loss_dict['evaluation_metrics_kid'] = kid
        #     loss_dict['evaluation_metrics_g_diversity'] = g_diversity

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        foot_contact_loss_val = loss_reduced["foot_contact"].mean().item()
        encourage_contact_loss_val = loss_reduced["encourage_contact"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        if args.action_recog_model is not None and args.entity != 'Edge':
            fid_metric = loss_reduced['evaluation_metrics_fid'].mean().item()
            kid_metric = loss_reduced['evaluation_metrics_kid'].mean().item()
            g_diversity_metric = loss_reduced['evaluation_metrics_g_diversity'].mean().item()

        if get_rank() == 0:
            description_str = f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; " + \
                              f"path: {path_loss_val:.4f}; " + \
                              f"foot contact: {foot_contact_loss_val:.4f}; " + \
                              f"encourage contact: {encourage_contact_loss_val:.4f}; " + \
                              f"mean path: {mean_path_length_avg:.4f}; "
            if isinstance(pbar, tqdm):
                pbar.set_description(description_str)
            elif i % 100 == 0:
                print(f'[{i}/{args.iter}]', description_str)

            if args.clearml or args.tensorboard:
                for loss_name, loss_val in zip(['Generator', 'Discriminator', 'R1', 'Path', 'Foot', 'Encourage contact'],
                                               [g_loss_val, d_loss_val, r1_val, path_loss_val, foot_contact_loss_val, encourage_contact_loss_val]):
                    logger.report_scalar("Losses", loss_name, iteration=i, value=loss_val)
                if args.action_recog_model is not None:
                    for metric_name, metric_val in zip(['FID', 'KID', 'Diversity'], [fid_metric, kid_metric, g_diversity_metric]):
                        logger.report_scalar("Evaluation metrics", metric_name, iteration=i, value=metric_val)

            if i == 0 or (i+1) % report_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "mean_joints": mean_joints,
                        "std_joints": std_joints
                    },
                    osp.join(args.model_save_path, f"checkpoint/{str(i).zfill(6)}.pt")
                )
                fake_motion = fake_img.transpose(1,2).detach().cpu().numpy()

                motion_path = osp.join(animations_output_folder, 'fake_motion_{}.bvh'.format(i))
                motion2bvh(fake_motion[0], motion_path, parents=entity.parents_list, entity=entity.str(), edge_rot_dict_general=edge_rot_dict_general)
                if args.clearml:
                    logger.report_media(title='Animation', series='Predicted Motion', iteration=i, local_path=motion_path)

                fig = motion2fig(fake_motion, H=512, W=512, entity=entity.str(),
                                 edge_rot_dict_general=edge_rot_dict_general)
                fig_path = osp.join(images_output_folder, 'fake_motion_{}.png'.format(i))
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()  # close figure
                if args.clearml:
                    logger.report_media(title='Image', series='Predicted Motion', iteration=i, local_path=fig_path)

                if i != 0:
                    torch.cuda.synchronize()
                    end_time_measure = time.time()
                    elapsed = end_time_measure - start_time_measure
                    time_measure.append(elapsed)
                    print(f'\nTime of last {report_every} iterations: {int(elapsed)} seconds.')
                    start_time_measure = time.time()
    mean_times = sum(time_measure)/len(time_measure)
    print(f'\nAverage time for {report_every} iterations: {mean_times} seconds.')


def calc_evaluation_metrics(args, device, g_ema, entity, std_joints, mean_joints):
    # create stgcn model
    stgcn_model = evaluate.initialize_model(device, modelpath= args.action_recog_model, dataset = args.dataset)

    # generate motions
    generated_motions = evaluate.generate(args, g_ema, device, mean_joints, std_joints, entity=entity)
    generated_motions = generated_motions[:, :15]
    generated_motions -= generated_motions[:, 8:9, :, :]
    iterator_generated = data.DataLoader(generated_motions, batch_size=500, shuffle=False, num_workers=8)

    # get features with stgcn
    generated_features, generated_predictions = evaluate.compute_features(stgcn_model, iterator_generated)
    generated_stats = evaluate.calculate_activation_statistics(generated_features)

    # load gt dataset and get features
    gt_motion_data_eval = np.load(args.act_rec_gt_path, allow_pickle=True)
    gt_motion_data_eval = gt_motion_data_eval[:, :15]
    gt_motion_data_eval -= gt_motion_data_eval[:, 8:9, :, :]  # locate root joint of all frames at origin
    iterator_dataset = data.DataLoader(gt_motion_data_eval, batch_size=64, shuffle=False, num_workers=8)
    dataset_features, dataset_predictions = evaluate.compute_features(stgcn_model, iterator_dataset)
    real_stats = evaluate.calculate_activation_statistics(dataset_features)
    # compute metrics
    #fid
    fid = evaluate.calculate_fid(generated_stats, real_stats)
    #kid
    kid = evaluate.calculate_kid(dataset_features.cpu(), generated_features.cpu())
    #generated diversity
    g_diversity = evaluate.calculate_diversity(generated_features)
    # precision/recall (lower priority)

    return fid, kid[0], g_diversity

def main(args_not_parsed):
    parser = TrainOptions()
    args = parser.parse_args(args_not_parsed)
    device = args.device
    traits_class = setup_env(args, get_traits=True)

    if args.clearml:
        output_folder = osp.expanduser('~/train_outputs')
        os.makedirs(output_folder, exist_ok=True)

        task = Task.init(project_name='stylegan2_motion_skeleton',
                         task_name=args.name,  # 'Jasper_all_5K_no_norm_mixing_0p9_conv3_fan_in_revw',
                         output_uri=output_folder)
        logger = task.get_logger()
        task_destination = task._get_output_destination_suffix()
        # task_destination =  re.sub('(.*\.[a-f\d]{3})[a-f\d]+([a-f\d]{3})', '\g<1>_\g<2>', task_destination)
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
        output_folder = args.model_save_path if args.model_save_path is not None else osp.expanduser('~/tmp')
        logger = None
        images_output_folder = osp.join(output_folder, 'images')
        animations_output_folder = osp.join(output_folder, 'animations')

    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(animations_output_folder, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(osp.join(args.model_save_path, 'checkpoint'), exist_ok=True)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0
    entity = eval(args.entity)

    generator = Generator(
        args.latent, args.n_mlp, traits_class=traits_class, entity=entity, n_inplace_conv=args.n_inplace_conv
    ).to(device)
    discriminator = Discriminator(traits_class=traits_class, entity=entity, n_inplace_conv=args.n_inplace_conv
    ).to(device)
    g_ema = Generator(
        args.latent, args.n_mlp, traits_class=traits_class, entity=entity, n_inplace_conv=args.n_inplace_conv
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.g_lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.d_lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    motion_data_raw = np.load(args.path, allow_pickle=True)
    motion_data, mean_joints, std_joints, edge_rot_dict_general = motion_from_raw(args, motion_data_raw)

    gt_bone_lengths = calc_bone_lengths(motion_data) if args.entity == 'Joint' else None

    motion_path = osp.join(animations_output_folder, 'real_motion.bvh')
    motion2bvh(motion_data[0], motion_path, parents=entity.parents_list, entity=args.entity,
               edge_rot_dict_general=edge_rot_dict_general)
    if args.clearml:
        logger.report_media(title='Animation', series='Ground Truth Motion', iteration=0, local_path=motion_path)

    fig = motion2fig(motion_data, H=512, W=512, entity=args.entity,
                     edge_rot_dict_general=edge_rot_dict_general)
    fig_name = osp.join(images_output_folder, 'real_motion.png')
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close() # close figure
    if args.clearml:
        logger.report_media(title='Image', series='Ground Truth Motion', iteration=0, local_path=fig_name)

    motions_data_torch = torch.from_numpy(motion_data)
    dataset = TensorDataset(motions_data_torch)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, logger, entity,
          animations_output_folder, images_output_folder, mean_joints, std_joints, gt_bone_lengths, edge_rot_dict_general)


if __name__ == "__main__":
    main(sys.argv[1:])