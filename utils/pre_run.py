import argparse
import re
import numpy as np
import torch
from models.gan import Generator, Discriminator
from utils.data import Edge, motion_from_raw

# region Parser Options
class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", type=str, default="cuda")
        self.parser = parser

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
        parser.add_argument("--tensorboard", action="store_true", help="use tensorboard for loss recording")
        parser.add_argument("--model_save_path", type=str, default='checkpoint', help="path for saving model")
        parser.add_argument("--on_cluster_training", action='store_true',
                            help="When training on cluster, use standard print instead of tqdm")
        parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
        parser.add_argument("--dataset", type=str, default='mixamo', help="mixamo or humanact12")
        parser.add_argument("--iter", type=int, default=80000, help="total training iterations")
        parser.add_argument("--report_every", type=int, default=2000, help="number of iterations between saving model checkpoints")
        parser.add_argument("--augment_p", type=float, default=0,
                            help="probability of applying augmentation. 0 = use adaptive augmentation")
        parser.add_argument("--action_recog_model", type=str,
                            default='evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar',
                            help="pretrained action recognition model used for feature extraction when computing evaluation metrics FID, KID, diversity")


class TrainOptions(TrainBaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        parser = self.parser
        parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        parser.add_argument("--path_regularize",type=float,default=2,
                            help="weight of the path length regularization",)
        parser.add_argument("--path_batch_shrink",type=int,default=2,
                            help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        parser.add_argument("--g_foot_reg_weight", type=float, default=1,
                            help="weight of the foot contact regularization")
        parser.add_argument("--g_encourage_contact_weight", type=float, default=0.01,
                            help="weight of the foot contact encouraging regularization")
        parser.add_argument("--g_reg_every",type=int,default=4, help="interval of the applying path length regularization",)
        parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training",)
        parser.add_argument("--g_lr", type=float, default=0.002, help="generator learning rate")
        parser.add_argument("--channel_multiplier", type=int, default=2,
                            help="channel multiplier factor for the model. config-f = 2, else = 1",)
        parser.add_argument("--name", type=str, default="no_name_defined",
                            help="name to be used for clearml experiment. example: Jasper_all_5K_no_norm_mixing_0p9_conv3_fan_in")
        parser.add_argument("--normalize", action="store_true", help="normalize data")
        parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        parser.add_argument("--skeleton", action="store_true", help="use skeleton-aware architecture")
        parser.add_argument("--joints_pool", action="store_true",
                            help="manipulate joints by pool/unpool rather than conv (skeleton-aware only)")
        parser.add_argument("--conv3", action="store_true", help="use 3D convolutions (skeleton-aware only)")
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
        assert (not (args.skeleton | args.conv3 | args.joints_pool)) | args.skeleton & (args.conv3 ^ args.joints_pool)
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
        self.parser = parser


class OptimOptions(TestBaseOptions):
    def __init__(self):
        super(OptimOptions, self).__init__()
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
        parser.add_argument('--text_path', type=str, help='Path to texts to generate', default=None)

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
        parser.add_argument('--fast', action='store_true', help='skip metrics that require long evaluation')
        parser.add_argument('--actor_motions_path', type=str, help='path to randomly generated actor motions')


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
    network_args_unique = ['skeleton', 'entity', 'glob_pos', 'joints_pool', 'conv3', 'foot', 'normalize',
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
        return list(range(int(m.group(1)), int(m.group(2))+1))
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
    if args.glob_pos:
        # add global position adjacency nodes to neighbouring lists
        # this method must be called BEFORE the Generator and the Discriminator are initialized
        Edge.enable_global_position()

    if 'Edge' in args.entity and args.rotation_repr == 'repr6d':
        Edge.enable_repr6d()

    if args.foot:
        Edge.enable_foot_contact()
        args.axis_up = 1

    if get_traits:
        from utils.traits import SkeletonAwarePoolTraits, SkeletonAwareConv3DTraits, NonSkeletonAwareTraits
        if args.skeleton:
            if args.joints_pool:
                traits_class = SkeletonAwarePoolTraits
            elif args.conv3:
                traits_class = SkeletonAwareConv3DTraits
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

    entity = eval(args.entity)

    g_ema = Generator(
        args.latent, args.n_mlp, traits_class=traits_class, entity=entity, n_inplace_conv=args.n_inplace_conv
    ).to(args.device)

    g_ema.load_state_dict(checkpoint["g_ema"])

    discriminator = Discriminator(traits_class=traits_class, entity=entity, n_inplace_conv=args.n_inplace_conv
                                  ).to(args.device)

    discriminator.load_state_dict(checkpoint["d"])

    mean_joints = checkpoint['mean_joints']
    std_joints = checkpoint['std_joints']


    if return_motion_data:
        motion_data_raw = np.load(args.path, allow_pickle=True)
        motion_data, mean_joints, std_joints, edge_rot_dict_general = motion_from_raw(args, motion_data_raw)

        mean_latent = g_ema.mean_latent(args.truncation_mean)

        return g_ema, discriminator, motion_data, mean_latent, edge_rot_dict_general

    return g_ema, discriminator, checkpoint, entity, mean_joints, std_joints
