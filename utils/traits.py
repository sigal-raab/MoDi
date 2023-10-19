import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from models.gan import Upsample
from motion_class import StaticMotionOneHierarchyLevel
from Motion.AnimationStructure import children_list
from models.skeleton import SkeletonPool, SkeletonUnpool


def neighbors_by_distance(layer: StaticMotionOneHierarchyLevel, dist=1):
    assert dist in [0, 1], 'distance larger than 1 is not supported yet'

    if dist == 0:  # code should be general to any distance. for now dist==1 is the largest supported
        number_of_joints = len(layer.parents) + layer.use_global_position + len(layer.feet_indices)
        return {joint_idx: [joint_idx] for joint_idx in range(number_of_joints)}

    children = children_list(layer.parents)
    neighbors = {joint: [joint] + ([layer.parents[joint]] if layer.parents[joint] != -1 else []) + children[joint].tolist() for joint in range(len(layer.parents))}

    # global position should have same neighbors of root and should become his neighbors' neighbor
    if layer.use_global_position:
        root_index = layer.parents.index(-1)
        global_position_index = len(layer.parents)

        neighbors[global_position_index] = [global_position_index] + neighbors[root_index]

        for root_neighbor in neighbors[root_index].copy():
            neighbors[root_neighbor].append(global_position_index)

    # 'contact' joint should have same neighbors of related joint and should become his neighbors' neighbor
    for foot_index, foot_contact_index in zip(layer.feet_indices, range(len(neighbors), len(neighbors) + len(layer.feet_indices))):
        neighbors[foot_contact_index] = [foot_contact_index, foot_index]

        neighbors[foot_index].append(foot_contact_index)

    return neighbors



class SkeletonTraits(nn.Module):
    # static variables
    channel_multiplier = 2
    n_channels_max = [256, 128, 64, 32 * channel_multiplier, 16 * channel_multiplier]
    num_frames = 64

    @classmethod
    def set_num_frames(cls, n_f):
        cls.num_frames = n_f
        if n_f == 128:
            cls.n_channels_max = [256, 128, 64, 32, 16 * SkeletonTraits.channel_multiplier, 8 * SkeletonTraits.channel_multiplier]

    def __init__(self, num_frames=64):
        super().__init__()

        self.transposed_conv_func= F.conv_transpose2d
        self.conv_func = F.conv2d

    def upsample(self, blur_kernel):
        return Upsample(blur_kernel, skeleton_traits=self)

    def updown_pad(self, kernel_size=None):
        return (0,0)

    @staticmethod
    def skeleton_aware():
        raise 'not implemented in base class'

    def fixed_dim_pad(self, kernel_size):
        raise 'not implemented in base class'

    def upfirdn_pad(self, pad_before_data, pad_after_data):
        raise 'not implemented in base class'

    def upfirdn_updown(self, up, down):
        raise 'not implemented in base class'

    def upfirdn_kernel_exp(self, up, down):
        raise 'not implemented in base class'

    def blur_pad(self):
        raise 'not implemented in base class'

    def out_channel_expanded(self, out_channel):
        return out_channel

    def kernel_height(self, kernel_size):
        raise 'not implemented in base class'

    def weight_internal(self, in_channel, out_channel, kernel_size):
         return torch.randn(self.out_channel_expanded(out_channel), in_channel,
                            self.kernel_height(kernel_size), kernel_size)

    def weight(self, in_channel, out_channel, kernel_size, modulation=False):
        weight = self.weight_internal(in_channel, out_channel, kernel_size)
        if modulation:
            #  the weird '1' at the front is a preparations for batch instances where each of them is weight
            #  multiplied by a scale that is specific to one instance
            weight = weight.unsqueeze(0)
        return nn.Parameter(weight)

    def flip_if_needed(self, weight):
        return weight

    def mask(self, weight, out_channel, kernel_size):
        mask = self.mask_internal(weight, out_channel, kernel_size)  # torch.ones_like(weight) #
        return nn.Parameter(mask, requires_grad=False)

    def mask_internal(self, weight, out_channel, kernel_size):
        raise 'not implemented in base class'

    def reshape_style(self, style):
        return style

    # we run demod over all axses that are not batch or out_channel.
    # for non conv3 configuration those channels would be in_channel, kernel_hight(in_j), kernel_width,
    def norm_axis(self, weight):
        ndim = weight.ndim
        return list(range(ndim-3,ndim))

    def reshape_input_before_transposed_conv(self, input, batch, width):
        raise 'not implemented in base class'

    def reshape_input_before_conv(self, input, batch, width):
        raise 'not implemented in base class'

    def reshape_output_after_conv(self, output):
        return output

    def reshape_1D_kernel(self, kernel):
        raise 'not implemented in base class'

    def reshape_weight_before_transposed_conv(self, weight, batch, in_channel, out_channel):
        return weight

    def blur(self, blur_func, out):
        return blur_func(out)

    @staticmethod
    def kernel_dim():
        return 2

    @staticmethod
    def n_joints(entity):
        raise 'not implemented in base class'

    @classmethod
    def n_levels(cls, *args):
        raise 'not implemented in base class'

    @classmethod
    def n_frames(cls, entity):
        if cls.num_frames == 64:
            n_frames_max = [4, 8, 16, 32, 64]
        elif cls.num_frames == 128:
            n_frames_max = [4, 8, 16, 32, 64, 128]
        n_levels = cls.n_levels(entity)
        assert n_levels <= len(n_frames_max)
        return n_frames_max[-n_levels:]

    @classmethod
    def n_channels(cls, entity):
        n_channels_max = cls.n_channels_max
        n_levels = cls.n_levels(entity)
        assert n_levels <= len(n_channels_max)
        return n_channels_max[-n_levels:]

    @staticmethod
    def is_pool():
        return False


class NonSkeletonAwareTraits(SkeletonTraits):
    def __init__(self, layer: StaticMotionOneHierarchyLevel):
        super().__init__()

        self.updown_stride = (2, 2)

        # ths following are not really needed for a non-skeleton-aware class.
        # they are kept for compitability with the other classes
        self.layer = layer
        self.pooling_list = layer.pooling_list
        self.parents = layer.parents
        self.larger_n_joints = layer.edges_number
        self.smaller_n_joints = layer.edges_number_after_pooling
        self.upfirdn_kernel_exp = 2
        self.need_blur = True

    @classmethod
    def n_levels(cls, *args):
        return len(cls.n_channels_max)

    @staticmethod
    def skeleton_aware():
        return False

    def fixed_dim_pad(self, kernel_size):
        return kernel_size // 2

    def upfirdn_pad(self, pad_before_data, pad_after_data):
        return (pad_before_data, pad_after_data, pad_before_data, pad_after_data)

    def upfirdn_updown(self, up, down):
        return (up, up, down, down)

    def kernel_height(self, kernel_size):
        return kernel_size

    def mask_internal(self, weight, out_channel, kernel_size):
        return torch.ones_like(weight)

    def reshape_input_before_transposed_conv(self, input, batch, width):
        return input

    def reshape_input_before_conv(self, input, batch, width):
        return input

    def reshape_1D_kernel(self, kernel):
        return kernel[None, :] * kernel[:, None]

    @staticmethod
    def n_joints(_):
        return [1, 2, 4, 8, 16]


class SkeletonAwareTraits(SkeletonTraits):
    def __init__(self, layer: StaticMotionOneHierarchyLevel):
        super().__init__()

        self.layer = layer
        self.parents = layer.parents
        self.pooling_list = layer.pooling_list
        self.updown_stride = (1, 2)
        self.larger_n_joints = layer.edges_number
        self.smaller_n_joints = layer.edges_number_after_pooling
        self.upfirdn_kernel_exp = 1
        self.need_blur = True

    @staticmethod
    def skeleton_aware():
        return True

    def fixed_dim_pad(self, kernel_size):
        return (0, kernel_size // 2)

    def upfirdn_pad(self, pad_before_data, pad_after_data):
        # pad only rows, i.e., frames, because we up/down sample only frames using upfirdn
        return pad_before_data, pad_after_data, 0, 0

    def upfirdn_updown(self, up, down):
        return up, 1, down, 1  # up_y = down_y = 1, i.e., upsample and downsample scalas on the y (skeleton) axis are 1

    def kernel_height(self, kernel_size):
        return self.larger_n_joints

    def mask_internal(self, weight, out_channel, kernel_size):
        # mask = torch.ones_like(weight)
        # print('***************\nNO MASK\n**************')
        # return mask
        upsample = (self.larger_n_joints!=self.smaller_n_joints)
        neighbor_dist = -1
        if upsample:
            affectors_all_joint = self.pooling_list
        else:
            neighbor_dist = kernel_size // 2
            affectors_all_joint = neighbors_by_distance(self.layer, neighbor_dist)

        mask = torch.zeros_like(weight)
        for joint_idx, affectors_this_joint in affectors_all_joint.items():
            mask = self.mask_affectors(mask, out_channel, joint_idx, affectors_this_joint)
        return mask

    def mask_affectors(self, mask, out_channel, joint_idx, affectors_this_joint):
        assert out_channel * joint_idx < mask.shape[1] and all(
            [j < mask.shape[3] for j in affectors_this_joint])
        mask[..., out_channel * joint_idx: out_channel * (joint_idx + 1), :, affectors_this_joint, :] = 1
        return mask

    def reshape_1D_kernel(self, kernel):
        return kernel.unsqueeze(dim=0)

    @staticmethod
    def n_joints(motion_statics):
        return motion_statics.number_of_joints_in_hierarchical_levels()

    @classmethod
    def n_levels(cls, entity):
        return len(cls.n_joints(entity))


class SkeletonAwareConv3DTraits(SkeletonAwareTraits):
    def __init__(self, layer: StaticMotionOneHierarchyLevel):
        super().__init__(layer)

        self.updown_stride = (1,) + self.updown_stride
        self.transposed_conv_func = F.conv_transpose3d
        self.conv_func = F.conv3d

    def updown_pad(self, kernel_size=None):
        return (self.smaller_n_joints - 1,) + super().updown_pad(kernel_size)

    def fixed_dim_pad(self, kernel_size):
        return (self.smaller_n_joints - 1,) + super().fixed_dim_pad(kernel_size)

    def weight_internal(self, in_channel, out_channel, kernel_size):
        return torch.randn(out_channel, in_channel, self.smaller_n_joints, self.larger_n_joints, kernel_size)

    def mask_affectors(self, mask, out_channel, joint_idx, affectors_this_joint):
        mask[..., joint_idx, affectors_this_joint, :] = 1
        return mask

    def reshape_style(self, style):
        assert (style.view(style.shape[:3] + (1,) + style.shape[3:]) == style[:, :, :, np.newaxis]).all()
        return style[:, :, :, np.newaxis]  # add a dimension for out_j

    # conv3 dimensions are [batch, out_ch, in_ch, out_j, in_j, ker_wid]
    # demodulation should be ran over inch, in_j and ker_wid. the reason we don't run it on out_j is that
    # we have a separate set of weights for each output joints, so it's not that we multiply the full kernel
    # (i.e. [out_j, in_j, ker_wid]) by the data: we multiply only [1, in_j, ker_wid] by the data each time
    def norm_axis(self, weight):
        ndim = weight.ndim
        return [ndim-4, ndim-2, ndim-1]

    def reshape_input_before_transposed_conv(self, input, batch, width):
        return input.reshape(input.shape[:3] + (1, input.shape[-1]))  # add a dimension for out_j

    def reshape_input_before_conv(self, input, batch, width):
        return input.view(input.shape[:2] + (1,) + input.shape[2:])  # add a dimension for out_j

    def reshape_output_after_conv(self, output):
        assert output.shape[3] == 1
        return output.squeeze(3)

    @staticmethod
    def kernel_dim():
        return 3

    def flip_if_needed(self, weight):
        """ because of padding dim 2. see docx drawing """
        weight = torch.flip(weight, (2,))
        return weight


class SkeletonAwarePoolTraits(SkeletonAwareConv3DTraits):

    def __init__(self, layer: StaticMotionOneHierarchyLevel):
        super().__init__(layer)
        self.transposed_conv_func = self.transposed_conv_func2
        self.conv_func = self.conv_func2
        self.need_blur = False

    def upsample(self, blur_kernel):
        upsampling = 'bilinear'
        return nn.Upsample(scale_factor=(1, 2), mode=upsampling, align_corners=False)

    def updown_pad(self, kernel_size=None):
        return self.fixed_dim_pad(kernel_size)

    def blur(self, blur_func, out):
        return out

    def mask_internal(self, weight, out_channel, kernel_size):
        neighbor_dist = kernel_size // 2
        affectors_all_joint = neighbors_by_distance(self.layer, neighbor_dist)

        mask = torch.zeros_like(weight)
        for joint_idx, affectors_this_joint in affectors_all_joint.items():
            mask = self.mask_affectors(mask, out_channel, joint_idx, affectors_this_joint)
        return mask

    def fixed_dim_pad(self, kernel_size):
        return (self.larger_n_joints - 1,) + super().fixed_dim_pad(kernel_size)[1:]

    def weight_internal(self, in_channel, out_channel, kernel_size):
        return torch.randn(out_channel, in_channel, self.larger_n_joints, self.larger_n_joints, kernel_size)

    def transposed_conv_func2(self, input, weight, padding, groups, stride=1):
        upsampling = 'trilinear'
        upsample = nn.Upsample(scale_factor=(1,1,2), mode=upsampling, align_corners=False)
        unpool = SkeletonUnpool(self.pooling_list, output_joints_num=self.larger_n_joints)

        input = upsample(input)
        input = input.squeeze(3)
        input = unpool(input)
        input = input.unsqueeze(2)
        input = F.conv3d(input, weight, padding=padding, groups=groups)

        # input = input[:, :-1, :]
        input = self.reshape_output_after_conv(input)
        return input

    def reshape_weight_before_transposed_conv(self, weight, batch, in_channel, out_channel):
        weight = weight.reshape((batch, in_channel, out_channel) + weight.shape[2:]).transpose(1, 2)
        weight = weight.reshape((batch * out_channel, in_channel) + weight.shape[3:])
        return weight

    def conv_func2(self, input, weight, padding, groups=1, stride=1, bias = None):
        input = F.conv3d(input, weight, padding=padding, groups=groups, stride=stride)
        input = self.reshape_output_after_conv(input)
        if (self.smaller_n_joints != self.larger_n_joints):
            pool = SkeletonPool(self.pooling_list, pooling_mode='mean', input_joints_num=self.larger_n_joints)
            input = pool(input)
        input = input.unsqueeze(3)
        return input

    @staticmethod
    def is_pool():
        return True


class SkeletonAwareFastConvTraits(SkeletonAwareConv3DTraits):
    def __init__(self, layer: StaticMotionOneHierarchyLevel):
        super().__init__(layer)
        self.updown_stride = self.updown_stride[1:]

        def conv_func(inputs, weight, padding=0, stride=1, groups=1, **kwargs):
            batch = inputs.shape[0]
            out_channels = weight.shape[0]

            weight = weight.transpose(1, 2).flatten(start_dim=0, end_dim=1)

            out = F.conv2d(inputs, weight, padding=padding[1:], stride=stride, groups=groups, **kwargs)
            out = out.reshape(batch, out_channels, -1, out.shape[-1])

            return out
        self.conv_func = conv_func

        def transposed_conv_func(inputs, weight, padding, stride, groups, **kwargs):
            out_channels = weight.shape[1]

            weight = weight.flip(2).transpose(2, 3).flatten(start_dim=1, end_dim=2)
            # original weights are [C_in, C_out, E_in, E_out, k] and we switch to [C_in, C_out * E_out^*, E_in, k]

            out = F.conv_transpose2d(inputs, weight, padding=padding[:-1], stride=stride, groups=groups, **kwargs)
            out = out.reshape(1, groups * out_channels, -1, out.shape[-1])

            return out

        self.transposed_conv_func = transposed_conv_func

    def reshape_input_before_transposed_conv(self, inputs, batch, width):
        return inputs

    def reshape_input_before_conv(self, inputs, batch, width):
        return inputs

    def reshape_output_after_conv(self, output):
        return output

    def flip_if_needed(self, weight):
        return weight
