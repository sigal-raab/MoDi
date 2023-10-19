import torch

from motion_class import StaticData
from models.kinematics import ForwardKinematicsJoint


class ConditionalMask:
    """
    This is a class used for create a masked input, for motion completion or motion inbetween
    """
    def __init__(self, args, n_frames, keep_loc, keep_rot, normalisation_data=None, noise_level=0.):
        """
        Negative n_frames for in between
        """
        self.pos_offset = -3 if args.foot else -1 # global position idx
        self.rot_offset = 0 # pelvis idx rotations
        self.noise_level = noise_level

        # todo: remove the std mean related code when releasing
        if normalisation_data is not None:
            self.std, self.mean = normalisation_data['std'], normalisation_data['mean']
            self.std = torch.tensor(self.std, dtype=torch.float32)
            self.mean = torch.tensor(self.mean, dtype=torch.float32)
        else:
            self.std, self.mean = None, None

        if n_frames == 0:
            self.func = 'inversion'
        elif n_frames > 0:
            self.func = 'mask'
            self.n_frames = n_frames
            self.keep_loc = keep_loc
            self.keep_rot = keep_rot
        else:
            self.func = 'inbetween'
            self.n_frames = -n_frames
            self.keep_loc = 0
            self.keep_rot = 0

    def __call__(self, motion, indicator_only=False, cond_length=None):
        if self.noise_level > 0:
            motion = motion + torch.randn_like(motion) * self.noise_level

        if self.func == 'inversion':
            return motion
        else:
            res = torch.zeros_like(motion)

        if cond_length is not None:
            if cond_length != self.n_frames:
                print('Warning: cond_length != self.n_frames')
                n_frames = cond_length

        indicator = torch.zeros_like(motion[:, :1, ...])
        if self.func == 'mask':
            sli = slice(n_frames)
            res[..., sli] = motion[..., sli]
            indicator[..., sli] = 1
            if self.keep_loc:
                res[:, :, self.pos_offset] = motion[:, :, self.pos_offset]
            if self.keep_rot:
                res[:, :, self.rot_offset] = motion[:, :, self.rot_offset]
        elif self.func == 'inbetween':
            for sli in (slice(0, n_frames), slice(-n_frames, None)):
                res[..., sli] = motion[..., sli]
                indicator[..., sli] = 1

        if indicator_only:
            return indicator

        return res

class GlobalPosLoss:
    def __init__(self, args):
        self.pos_offset = -3 if args.foot else -1 # global position idx
        self.rot_offset = 0 # pelvis idx rotations

        if args.loss_type == 'L1':
            self.loss = torch.nn.L1Loss()
        elif args.loss_type == 'L2':
            self.loss = torch.nn.MSELoss()

    def __call__(self, pred, target):
        return self.loss(pred[:, :, self.pos_offset], target[:, :, self.pos_offset]) + self.loss(pred[:, :, self.rot_offset], target[:, :, self.rot_offset])

class ReconLoss:
    def __init__(self, loss_type):
        if loss_type == 'L1':
            self.loss = torch.nn.L1Loss()
        elif loss_type == 'L2':
            self.loss = torch.nn.MSELoss()

    def __call__(self, pred, target):
        return self.loss(pred, target)


def sigmoid_for_contact(predicted_foot_contact):
    return torch.sigmoid((predicted_foot_contact - 0.5) * 2 * 6)


class ContactLabelLoss:
    def __init__(self):
        self.contact_offset = [-2, -1]
        self.criteria = torch.nn.BCELoss()

    def get_contact_label(self, motion):
        label = motion[:, 0, self.contact_offset]
        return label

    def __call__(self, pred, target):
        return self.criteria(sigmoid_for_contact(self.get_contact_label(pred)), self.get_contact_label(target.detach()))


class PositionLoss:
    def __init__(self, motion_statics : StaticData, use_glob_pos, use_contact, use_velocity,
                 mean_joints, std_joints, local_frame=False):
        self.use_glob_pos = use_glob_pos
        self.fk = ForwardKinematicsJoint(motion_statics .parents, torch.from_numpy(motion_statics .offsets).to('cuda'))
        self.criteria = torch.nn.MSELoss()
        self.local_frame = local_frame
        self.pos_offset = -3 if use_contact else -1
        self.use_velocity = use_velocity

        self.mean_joints = mean_joints
        self.std_joints = std_joints

    def get_pos(self, motion_data):
        motion_data = motion_data * self.std_joints[:, :, :motion_data.shape[2]] + \
                      self.mean_joints[:, :, :motion_data.shape[2]]
        motion_for_fk = motion_data.transpose(1,
                                              3)  # samples x features x joints x frames  ==>  samples x frames x joints x features
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


class PositionLossRoot:
    def __init__(self, motion_statics , device, use_glob_pos, use_contact, use_velocity,
                 mean_joints, std_joints, local_frame=False):
        self.use_glob_pos = use_glob_pos
        self.fk = ForwardKinematicsJoint(motion_statics .parents, motion_statics .offsets)
        self.criteria = torch.nn.MSELoss()
        self.local_frame = local_frame
        self.pos_offset = -3 if use_contact else -1
        self.use_velocity = use_velocity

        self.mean_joints = mean_joints
        self.std_joints = std_joints

    def get_pos(self, motion_data):
        motion_data = motion_data * self.std_joints[:, :, :motion_data.shape[2]] + \
                      self.mean_joints[:, :, :motion_data.shape[2]]
        motion_for_fk = motion_data.transpose(1, 3)  # samples x features x joints x frames  ==>  samples x frames x joints x features
        if self.use_glob_pos:
            #  last 'joint' is global position. use only first 3 features out of it.
            glob_pos = motion_for_fk[:, :, self.pos_offset, :3]
            if self.use_velocity:
                glob_pos = torch.cumsum(glob_pos, dim=1)
                return glob_pos

        return glob_pos.fill_(0.)

    def __call__(self, pred, target):
        pred = self.get_pos(pred)
        target = self.get_pos(target)
        return self.criteria(pred, target)
