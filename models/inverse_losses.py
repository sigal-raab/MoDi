import torch
from train import g_foot_contact_loss_v2
import numpy as np
import torch.nn.functional as F

from motion_class import StaticData
from models.kinematics import ForwardKinematicsJoint


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, args, discriminator):
        super(DiscriminatorLoss, self).__init__()
        self.args = args
        self.discriminator = discriminator

    def forward(self, input):
        fake_pred = self.discriminator(input)
        return F.softplus(-fake_pred).mean()


class LatentCenterRegularizer(torch.nn.Module):
    def __init__(self, args, latent_center):
        super(LatentCenterRegularizer, self).__init__()
        self.args = args
        self.latent_center = latent_center

    def forward(self, input):
        return F.mse_loss(input, self.latent_center)


class PositionLoss:
    def __init__(self, motion_statics: StaticData, normalisation_data, device, use_glob_pos, use_contact, use_velocity, local_frame=False):
        offsets = motion_statics.offsets
        offsets = torch.from_numpy(offsets).to(device).type(torch.float32)
        self.use_glob_pos = use_glob_pos
        self.use_velocity = use_velocity

        self.fk = ForwardKinematicsJoint(motion_statics.parents, offsets)
        self.normalisation_data = normalisation_data
        self.criteria = torch.nn.MSELoss()
        self.local_frame = local_frame
        if use_contact:
            self.pos_offset = -3 if use_contact else -1

    def get_pos(self, motion_data):
        motion_data = motion_data * self.normalisation_data['std'][:, :, :motion_data.shape[2]] + \
                      self.normalisation_data['mean'][:, :, :motion_data.shape[2]]
        # samples x features x joints x frames  ==>  samples x frames x joints x features
        motion_for_fk = motion_data.transpose(1, 3)

        if self.use_glob_pos:
            #  last 'joint' is global position. use only first 3 features out of it.
            glob_pos = motion_for_fk[:, :, self.pos_offset, :3]
            if self.use_velocity:
                glob_pos = torch.cumsum(glob_pos, dim=1)
            if self.local_frame:
                glob_pos.fill_(0.)
            motion_for_fk = motion_for_fk[:, :, :self.pos_offset]

        joint_location = self.fk.forward_edge_rot(motion_for_fk, glob_pos)
        return joint_location

    def __call__(self, pred, target):
        pred = self.get_pos(pred)
        target = self.get_pos(target)
        return self.criteria(pred, target)


class FootContactUnsupervisedLoss(torch.nn.Module):
    def __init__(self, motion_statics: StaticData, normalisation_data: {str: torch.Tensor},
                 use_global_position: bool, use_velocity: bool):
        super(FootContactUnsupervisedLoss, self).__init__()

        self.motion_statics = motion_statics
        self.normalisation_data = normalisation_data
        self.use_global_position = use_global_position
        self.use_velocity = use_velocity

    def forward(self, motion):
        foot_contact = g_foot_contact_loss_v2(motion, self.motion_statics, self.normalisation_data,
                                              self.args.glob_pos, self.args.use_velocity)
        return foot_contact
