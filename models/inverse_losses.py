import torch
import numpy as np
import torch.nn.functional as F
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
    def __init__(self, edge_rot_dict_general, device, use_glob_pos, use_contact, local_frame=False):
        root_idx = 0
        offsets = np.insert(edge_rot_dict_general['offsets_no_root'], root_idx, edge_rot_dict_general['offset_root'],
                            axis=0)
        offsets = torch.from_numpy(offsets).to(device).type(torch.float32)
        self.use_glob_pos = use_glob_pos
        self.fk = ForwardKinematicsJoint(edge_rot_dict_general['parents_with_root'], offsets)
        self.edge_rot_dict_general = edge_rot_dict_general
        self.criteria = torch.nn.MSELoss()
        self.local_frame = local_frame
        if use_contact:
            self.pos_offset = -3 if use_contact else -1

    def get_pos(self, motion_data):
        edge_rot_dict_general = self.edge_rot_dict_general
        motion_data = motion_data * edge_rot_dict_general['std_tensor'][:, :, :motion_data.shape[2]] + \
                      edge_rot_dict_general['mean_tensor'][:, :, :motion_data.shape[2]]
        motion_for_fk = motion_data.transpose(1,
                                              3)  # samples x features x joints x frames  ==>  samples x frames x joints x features
        if self.use_glob_pos:
            #  last 'joint' is global position. use only first 3 features out of it.
            glob_pos = motion_for_fk[:, :, self.pos_offset, :3]
            if 'use_velocity' in edge_rot_dict_general and edge_rot_dict_general['use_velocity']:
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
