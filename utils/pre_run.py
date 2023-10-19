import argparse
import re
import numpy as np
import torch

from models.gan import Generator, Discriminator
from utils.data import Joint, motion_from_raw
from motion_class import StaticData


# region Parser Options
class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.parser = parser
        parser.add_argument("--device", type=str, default="cuda")

    def parse_args(self, args=None):
        return self.after_parse(self.parser.parse_args(args))

    def after_parse(self, args):
        return args


class TrainBaseOptions(BaseOptions):
    def __init__(self):
        super(TrainBaseOptions, self).__init__()
        parser = self.parser
        parser.add_argument("--path", type=str, help="path to dataset")
        parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
        parser.add_argument("--d_lr", type=float, default=0.002, help="discriminator learning rate")
        parser.add_argument("--clearml", action="store_true", help="use trains logging")
        parser.add_argument("--name", type=str, default="no_name_defined",
                            help="name to be used for clearml experiment. example: Jasper_all_5K_no_norm_mixing_0p9_conv3_fan_in")
        parser.add_argument("--tensorboard", action="store_true", help="use tensorboard for loss recording")
        parser.add_argument("--model_save_path", type=str, default='checkpoint', help="path for saving model")
        parser.add_argument("--on_cluster_training", action='store_true',
                            help="When training on cluster, use standard print instead of tqdm")
        parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpu")
        parser.add_argument("--dataset", type=str, default='mixamo', help="mixamo or humanact12")
        parser.add_argument("--iter", type=int, default=80000, help="total training iterations")
        parser.add_argument("--report_every", type=int, default=2000, help="number of iterations between saving model checkpoints")
        parser.add_argument("--augment_p", type=float, default=0,
                            help="probability of applying augmentation. 0 = use adaptive augmentation")
        parser.add_argument("--action_recog_model", type=str,
                            help="pretrained action recognition model used for feature extraction when computing evaluation metrics FID, KID, diversity")
        parser.add_argument("--character", type=str, default='jasper', help='name of the character on the dataset')
        parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training",)


class TrainOptions(TrainBaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        parser = self.parser
        parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        parser.add_argument("--path_batch_shrink", type=int, default=2,
                            help="batch size reducing factor for the path length regularization (reduce memory consumption)", )
        parser.add_argument("--g_foot_reg_weight", type=float, default=1,
                            help="weight of the foot contact regularization")
        parser.add_argument("--g_encourage_contact_weight", type=float, default=0.01,
                            help="weight of the foot contact encouraging regularization")
        parser.add_argument("--g_reg_every",type=int,default=4, help="interval of the applying path length regularization",)
        parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        parser.add_argument("--g_lr", type=float, default=0.002, help="generator learning rate")
        parser.add_argument("--channel_multiplier", type=int, default=2,
                            help="channel multiplier factor for the model. config-f = 2, else = 1",)
        parser.add_argument("--normalize", action="store_true", help="normalize data")
        parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        parser.add_argument("--skeleton", action="store_true", help="use skeleton-aware architecture")
        parser.add_argument("--joints_pool", action="store_true",
                            help="manipulate joints by pool/unpool rather than conv (skeleton-aware only)")
        parser.add_argument("--conv3", action="store_true", help="use 3D convolutions (skeleton-aware only)")
        parser.add_argument("--conv3fast", action="store_true", help="use fast 2D convolutions (skeleton-aware only)")
        parser.add_argument("--entity", type=str, default='Edge', choices=['Joint', 'Edge'],
                            help="entity type: Joint for joint locations, or Edge for edge rotations")
        parser.add_argument("--glob_pos", action="store_true",
                            help="refrain from predicting global root position when predicting rotations")
        parser.add_argument('--return_sub_motions', action='store_true',
                            help='Return motions created by coarse pyramid levels')
        parser.add_argument("--foot", action="store_true", help="apply foot contact loss")
        parser.add_argument("--axis_up", choices=[0, 1, 2], type=int, default=1,
                            help="which axis points at the direction of a standing person's head? currently it is z for locations and y for rotations.")
        parser.add_argument("--v2_contact_loss", action='store_true', help="New contact loss")
        parser.add_argument("--use_velocity", action='store_true', help="Use velocity at root joint instead of position")
        parser.add_argument("--rotation_repr", type=str, default='quaternion')
        parser.add_argument('--latent', type=int, default=512, help='Size of latent space')
        parser.add_argument('--n_mlp', type=int, default=8, help='Number of MLP for mapping z to W')
        parser.add_argument('--n_frames_dataset', type=int, default=64)
        parser.add_argument("--n_inplace_conv", default=2, type=int,
                            help="Number of self convolutions within each hierarchical layer. StyleGAN original is 1. ")
        parser.add_argument('--act_rec_gt_path', type=str,
                            help='path to ground truth file that was used during action recognition train. Not needed unless is different from the one used by the synthesis network')
        self.parser = parser

    def after_parse(self, args):
        assert (not (args.skeleton | args.conv3 | args.joints_pool | args.conv3fast)) | args.skeleton & (args.conv3 ^ args.joints_pool ^ args.conv3fast)
        return args

class TrainEncoderOptions(TrainBaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument("--ckpt_existing", type=str, required=True, help='path to existing generative model')
        parser.add_argument("--n_frames", type=int, default=20, help='number of frames that is not masked')
        parser.add_argument("--keep_loc", type=int, default=0)
        parser.add_argument("--keep_rot", type=int, default=0)
        parser.add_argument("--n_latent_predict", type=int, default=1,
                            help='number of latent to predict, 1 for W space, bigger than 1 for Wplus spaces')
        parser.add_argument("--loss_type", type=str, default='L2')
        parser.add_argument("--overfitting", type=int, default=0)
        parser.add_argument("--lambda_pos", type=float, default=1.)
        parser.add_argument("--lambda_rec", type=float, default=10.)
        parser.add_argument("--lambda_contact", type=float, default=1000.)
        parser.add_argument("--lambda_global_pos", type=float, default=20.)
        parser.add_argument("--lambda_disc", type=float, default=0.)
        parser.add_argument("--lambda_reg", type=float, default=0.)
        parser.add_argument("--lambda_foot_contact", type=float, default=0.)
        parser.add_argument("--use_local_pos", type=int, default=1)
        # taken from train args
        parser.add_argument("--truncation_mean", type=int, default=4096,
                            help="number of vectors to calculate mean for the truncation")
        parser.add_argument("--encoder_latent_rec_idx", default=4,
                            help="which discriminator layer will be the one used for latent reconstruction?")
        parser.add_argument("--partial_loss", type=int, default=0)
        parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization for discrim.")
        parser.add_argument("--g_reg_every", type=int, default=4,
                            help="interval of the applying path length regularization")
        parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        parser.add_argument("--noise_level", type=float, default=0)
        parser.add_argument("--train_disc", type=int, default=0)
        parser.add_argument("--empty_disc", type=int, default=0)
        parser.add_argument("--partial_disc", type=int, default=0)
        parser.add_argument("--disc_freq", type=int, default=1)
        parser.add_argument("--train_with_generated", type=int, default=0)
        parser.add_argument("--use_half_rec_model", type=int, default=0)

    def after_parse(self, args):
        args = super().after_parse(args)
        if args.use_half_rec_model:
            assert args.action_recog_model == './evaluation/checkpoint_0300_no_globpos_32frames_acc_0.98.pth'
        return args


class TestBaseOptions(BaseOptions):
    def __init__(self):
        super(TestBaseOptions, self).__init__()
        parser = self.parser

        parser.add_argument("--motions", type=int, default=20, help="number of motions to be generated")
        parser.add_argument("--criteria", type=str, default='torch.nn.MSELoss')
        parser.add_argument('--path', type=str,
                            help='Path to ground truth file that was used during train. Not needed unless one wants to override the local path saved by the network')
        parser.add_argument('--out_path', type=str,
                            help='Path to output folder. If not provided, output folder will be <ckpt/ckpt_files/timestamp')
        parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
        parser.add_argument("--truncation_mean",type=int,default=4096,
                            help="number of vectors to calculate mean for the truncation",)
        parser.add_argument("--ckpt",type=str,help="path to the model checkpoint",)
        parser.add_argument("--simple_idx", type=int, default=0, help="use simple idx for output bvh files")
        parser.add_argument("--character", type=str, default='jasper', help='name of the character on the dataset')
        self.parser = parser

def _parse_list_nums(s):
    ''' accept comma seperated list of strings and return list of strings'''
    vals = s.split(',')
    return [int(v) for v in vals]


class TestEncoderOptions(TestBaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument("--full_eval", action='store_true')
        parser.add_argument("--ckpt_existing", type=str, required=False, help='path to existing generative model. Taken from the encoder checkpoint. Should have a value if want to use a different checkpoint than the one saved in the encoder')
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--eval_id", type=_parse_list_nums, help='list of idx for encoder. When using fusion application only first two indices apply.')
        parser.add_argument("--n_frames_override", type=int, help='number of frames that is not masked override encoder arguments')
        parser.add_argument("--application", type=str, default='inversion',
                            choices=['inversion', 'fusion', 'editing', 'editing_seed', 'denoising',
                                      'auto_regressive'])
        parser.add_argument("--n_frames_autoregressive", type=int, default=128)


class OptimOptions(TestBaseOptions):
    def __init__(self):
        super(OptimOptions, self).__init__()
        parser = self.parser
        self.parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
        self.parser.add_argument("--target_idx", default=0, type=int)
        self.parser.add_argument("--lambda_disc", default=0, type=float)
        self.parser.add_argument("--lambda_latent_center", default=0, type=float)
        self.parser.add_argument("--n_iters", default=5000, type=int)
        self.parser.add_argument("--lambda_pos", default=3e-3, type=float)
        self.parser.add_argument("--use_local_pos", default=1, type=int)
        self.parser.add_argument("--Wplus", type=int, default=1, help="Use Wplus space or not")


class GenerateOptions(TestBaseOptions):
    def __init__(self):
        super(GenerateOptions, self).__init__()
        parser = self.parser
        parser.add_argument("--type", type=str, default='sample',
                            choices=['sample', 'truncation_series', 'interp', 'edit'],
                            help="generation type: \n"
                                 "sample: generate n_motions motions\n"
                                 "interpolate: interpolate W space of two random motions \n"
                                 "edit: latent space editing\n")
        # related to sample
        parser.add_argument('--sample_seeds', type=_parse_num_range, help='Seeds to use for generation')
        parser.add_argument('--return_sub_motions', action='store_true',
                            help='Return motions created by coarse pyramid levels')
        parser.add_argument('--no_idle', action='store_true', help='sample only non-idle motions')

        # related to interpolate
        parser.add_argument('--interp_seeds', type=_parse_interp_seeds, help='Seeds to use for interpolation')

        # related to latent space editing
        parser.add_argument('--boundary_path', type=str, help='Path to boundary file')
        parser.add_argument('--edit_radius', type=float,
                            help='Editing radius (i.e., max change of W in editing direction)')


class EvaluateOptions(TestBaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument("--dataset", type=str, default='mixamo', choices=['mixamo', 'humanact12'],
                            help='on which dataset to evaluate')
        parser.add_argument("--rot_only", action="store_true",
                            help="refrain from predicting global root position when predicting rotations")
        parser.add_argument("--test_model", action="store_true", help="generate motions with model and evaluate")
        parser.add_argument("--test_actor", action="store_true", help="evaluate results from ACTOR model")

        parser.add_argument('--act_rec_gt_path', type=str,
                            help='path to ground truth file that was used during action recognition train. Not needed unless is different from the one used by the synthesis network')
        parser.add_argument('--actor_motions_path', type=str, help='path to randomly generated actor motions')
        parser.add_argument('--fast', action='store_true', help='skip metrics that require long evaluation')


class EditOptions(TestBaseOptions):
    def __init__(self):
        super(EditOptions, self).__init__()
        parser = self.parser
        parser.add_argument("--model_path", type=str, help="path to model file")
        parser.add_argument("--data_path", type=str, default=None,
                            help="path to data folder, if the 'generate_motions' stage has already been done")
        parser.add_argument("--score_path", type=str, default=None,
                            help="path to scores folder, if the 'calc_score' stage has already been done")
        parser.add_argument("--entity", type=str, default='Edge', choices=['Joint', 'Edge'],
                            help="entity type: joint for joint locations, or edge for edge rotations")
        parser.add_argument("--attr", type=str, default=['r_hand_lift_up'], nargs='+',
                            choices=['r_hand_lift_up', 'r_elbow_angle', 'r_wrist_accel', 'r_wrist_vert', 'verticality'],
                            help="list of attributes to be edited")

# endregion

def get_ckpt_args(args, loaded_args):
    network_args_unique = ['skeleton', 'entity', 'glob_pos', 'joints_pool', 'conv3', 'conv3fast', 'foot', 'normalize',
                           'axis_up', 'use_velocity', 'rotation_repr', 'latent',
                           'n_frames_dataset', 'n_inplace_conv', 'n_mlp', 'channel_multiplier']
    network_args_non_unique = ['path']
    for arg_name in network_args_unique + network_args_non_unique:

        # make sure network args don't coincide with command line args
        assert not hasattr(args, arg_name) or arg_name in network_args_non_unique, \
            f'{arg_name} is already defined in checkpoint and is not categorized "non-unique"'

        if arg_name in network_args_unique or getattr(args, arg_name, None) is None:
            arg_val = getattr(loaded_args, arg_name, None)
            setattr(args, arg_name, arg_val)
    return args

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]

def _parse_list_num_ranges(s):
    ''' accept comma seperated list of ranges 'a-c','d-e' and return list of lists of int [[a,b,c],[d,e]]'''
    ranges = s.split(',')
    return [_parse_num_range(r) for r in ranges]

def _parse_interp_seeds(s):
    ''' Accept comma seperated list of numbers or ranges 'a,b,c-d' and returns a list of lists [[a],[b],[c,d]]'''
    seeds = s.split(',')
    interps = []
    for seed in seeds:
        range_re = re.compile(r'^(\d+)-(\d+)$')
        m = range_re.match(seed)
        if m:
            interps.append([int(m.group(1)), int(m.group(2))])
        else:
            interps.append([int(seed)])
    return interps


def setup_env(args, get_traits=False):

    if get_traits:
        from utils.traits import SkeletonAwarePoolTraits, SkeletonAwareConv3DTraits,\
            NonSkeletonAwareTraits, SkeletonAwareFastConvTraits

        if args.skeleton:
            if args.joints_pool:
                traits_class = SkeletonAwarePoolTraits
            elif args.conv3:
                traits_class = SkeletonAwareConv3DTraits
            elif args.conv3fast:
                traits_class = SkeletonAwareFastConvTraits
            else:
                raise 'Traits cannot be selected.'
        else:
            traits_class = NonSkeletonAwareTraits

        if hasattr(args, 'n_frames_dataset') and args.n_frames_dataset == 128:
            traits_class.set_num_frames(128)
        return traits_class


def load_all_form_checkpoint(ckpt_path, args, return_motion_data=False):
    """Load everything from the path"""

    checkpoint = torch.load(ckpt_path)

    args = get_ckpt_args(args, checkpoint['args'])

    traits_class = setup_env(args, get_traits=True)

    motion_data_raw = np.load(args.path, allow_pickle=True)

    if args.entity == 'Edge':
        motion_statics = StaticData.init_from_motion(motion_data_raw[0], character_name=args.character,
                                             n_channels=4,
                                             enable_global_position=args.glob_pos,
                                             enable_foot_contact=args.foot,
                                             rotation_representation=args.rotation_repr)

    elif args.entity == 'Joint':
        motion_statics = StaticData.init_joint_static(Joint(), character_name=args.character)


    g_ema = Generator(
        args.latent, args.n_mlp, traits_class=traits_class, motion_statics=motion_statics, n_inplace_conv=args.n_inplace_conv
    ).to(args.device)

    g_ema.load_state_dict(checkpoint["g_ema"])

    discriminator = Discriminator(traits_class=traits_class, motion_statics=motion_statics, n_inplace_conv=args.n_inplace_conv
                                  ).to(args.device)

    discriminator.load_state_dict(checkpoint["d"])

    mean_joints = checkpoint['mean_joints']
    std_joints = checkpoint['std_joints']

    if return_motion_data:
        motion_data_raw = np.load(args.path, allow_pickle=True)
        motion_data, normalisation_data = motion_from_raw(args, motion_data_raw, motion_statics)
        mean_latent = g_ema.mean_latent(args.truncation_mean)

        return g_ema, discriminator, motion_data, mean_latent, motion_statics, normalisation_data, args

    return g_ema, discriminator, checkpoint, motion_statics, mean_joints, std_joints, args.entity, args
