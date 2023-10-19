##############################
#
# based on https://github.com/rosinality/stylegan2-pytorch
#
##############################

import math
import random
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

from models.skeleton import SkeletonUnpool
from motion_class import StaticData, StaticMotionOneHierarchyLevel


class LatentPrediction(nn.Module):
    def __init__(self, in_channel, latent_dim, n_latent, extra_linear=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_latent = n_latent
        out_dim = latent_dim * n_latent
        self.linear1 = EqualLinear(in_channel, out_dim, activation='fused_lrelu')
        if extra_linear:
            seq = []
            for i in range(extra_linear - 1):
                seq.append(EqualLinear(out_dim, out_dim, activation='fused_lrelu'))
            seq.append(EqualLinear(out_dim, out_dim))
            if len(seq) == 1:
                self.linear2 = seq[0]
            else:
                self.linear2 = nn.Sequential(*seq)
        else:
            self.linear2 = None

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)  # flatten. input may be already flat, depending on latent_rec_idx
        out = self.linear1(input)
        if self.linear2 is not None:
            out = self.linear2(out)
        out = out.reshape(input.shape[0], self.n_latent, self.latent_dim)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # divide each input array (separately) by the average of its squares (obtain average data norm of 1)
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k, skeleton_traits=None):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = skeleton_traits.reshape_1D_kernel(k)

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2, skeleton_traits=None):
        super().__init__()

        self.skeleton_traits = skeleton_traits
        self.factor = factor
        kernel = make_kernel(kernel, skeleton_traits) * (factor ** skeleton_traits.upfirdn_kernel_exp)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[1] - factor # originally this was shape[0]. However, for an image, shape[0]==shape[1] while for a skeleton we need shape[1]

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = skeleton_traits.upfirdn_pad(pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad, skeleton_traits=self.skeleton_traits)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2, skeleton_traits=None):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel, skeleton_traits)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, skeleton_traits=None):
        super().__init__()

        kernel = make_kernel(kernel, skeleton_traits)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** skeleton_traits.upfirdn_kernel_exp)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.skeleton_traits = skeleton_traits

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, skeleton_traits=self.skeleton_traits)

        return out


class EqualConv(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, skeleton_traits=None
    ):
        super().__init__()

        st = skeleton_traits
        self.weight = st.weight(in_channel, out_channel, kernel_size)
        self.mask = st.mask(self.weight, out_channel, kernel_size)

        norm_axis = st.norm_axis(self.weight)
        fan_in = self.mask.sum(norm_axis, keepdims=True)  # number of weights that are not zeroed, not considering output channels and output edges
        assert (fan_in[0] == fan_in).all()  # all output channels get same fan_in
        fan_in = fan_in[:1]  # hence we use only one output channel for clearness
        assert (fan_in <= np.prod(np.array(self.mask.shape)[norm_axis])).all()
        self.scale = nn.Parameter(1 / (fan_in**0.5), requires_grad=False)

        self.stride = stride
        self.padding = padding
        self.skeleton_traits = st

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None


    def forward(self, input):

        st = self.skeleton_traits
        batch, channels, height, width = input.shape
        input = st.reshape_input_before_conv(input, batch, width)
        weight = self.weight * self.mask
        weight = st.flip_if_needed(weight)
        out = st.conv_func(
            input,
            weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        out = st.reshape_output_after_conv(out)  # ==> batch, out_channel, out_height, out_width

        # i = torch.ones_like(input)
        # for ii in range(i.shape[-2]):  # each edge's value equals its index, for all samples, channels and frames
        #     ...: i[:, :, :, ii, :] = ii
        # w = torch.zeros_like(weight)
        # w[torch.flip(self.mask, (2,)) == 1] = 1
        # o = st.conv_func(i, w, bias=self.bias, stride=self.stride, padding=self.padding)
        # o = st.reshape_output_after_conv(o)

        return out


# this is like an nn.Linear layer, but with a scale that is related to the dimension and a learning rate
class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul  # 1/math.sqrt(in_dim) is equivalent to 1/(fan_in**0.5)
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        # ratio = 0.5 + 0.5 * self.negative_slope
        # mult = 1/ratio
        # print('ScaledLeakyReLU')
        # print([out.std(), (out * math.sqrt(2)).std()])
        # print([out.std(), (out * math.sqrt(mult)).std()])
        return out * math.sqrt(2) # multiply by sqrt(2) to get the std back to 1 (like in Kaiming He initialization)


class ModulatedConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        skeleton_traits=None,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.skeleton_traits = skeleton_traits
        self.out_channel_expanded = skeleton_traits.out_channel_expanded(out_channel)

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            pad = skeleton_traits.upfirdn_pad(pad0, pad1)

            self.blur = Blur(blur_kernel, pad, upsample_factor=factor, skeleton_traits=skeleton_traits)

        self.fixed_dim_pad = skeleton_traits.fixed_dim_pad(kernel_size)
        self.updown_pad = skeleton_traits.updown_pad(kernel_size)

        self.weight = skeleton_traits.weight(in_channel, out_channel, kernel_size, modulation=True)

        self.mask = skeleton_traits.mask(self.weight, out_channel, kernel_size)

        norm_axis = skeleton_traits.norm_axis(self.weight)

        # number of weights that are not zeroed, not considering output channels and output joints
        fan_in = self.mask.sum(norm_axis, keepdims=True)

        assert (fan_in[0, :] == fan_in[0, 0]).all()
        fan_in = fan_in[:, :1]
        # assert fan_in <= np.prod(self.mask.shape[skeleton_traits.norm_axis(self.weight)]) # in_channel x weight_volume
        self.scale = nn.Parameter(1/(fan_in**0.5), requires_grad=False)

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        if demodulate:
            self.demod_obj = 'data'
        else:
            self.demod_obj = None
        assert self.demod_obj in ['weights', 'data', None]

    def forward(self, input, style):
        # input is the learned constant on lowest level call, and layer features on later calls.
        # style is W[i].

        batch, in_channel, height, width = input.shape  # e.g.: [16, 256, 1, 4]

        # output size of modulation is [batch, in_channel]
        # A: Transform incoming W to style, i.e., std per feature.
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        style = self.skeleton_traits.reshape_style(style)

        weight = self.weight * self.mask

        # multiply weight by a different std for each instance and each channel in the batch.

        # scale: meaningless if demodulation is applied, because we multiply by scale and divide by
        # it in the demodulation line.
        #        * if demodulation is NOT applied, scale pushes the std of the output to be defined by style,
        #        and removes the effects of the increasing std that happens due to convolutions

        # multiply by fan_in and by predicted std
        # the scale multiplies the E dimension (2) and the style multiplies the channel one. (256)
        weight = self.scale * weight * style

        if self.demod_obj == 'weights':  # scale weights s.t. output features' std will be 1
            # scaling factor: 1 / sqrt(sum(w^2)) for kernel and channels
            demod = torch.rsqrt(weight.pow(2).sum(self.skeleton_traits.norm_axis(weight), keepdims=True) + 1e-8)
            weight = weight * demod

        # ==>  batch * out_channel_expanded, in_channel, kernel_height, kernel_width
        weight = weight.view(
            (weight.shape[0] * weight.shape[1],) + weight.shape[2:]
        )

        weight = self.skeleton_traits.flip_if_needed(weight)

        if self.upsample:
            # ==>  batch, in_channel, in_height,[ 1,] in_width
            input = self.skeleton_traits.reshape_input_before_transposed_conv(input, batch, width)

            # batches become channel-like, so they can 'catch' the mod/demod operation
            input = input.view((1, batch * input.shape[1]) + input.shape[2:])

            weight = weight.view(  # undo the view operation from before (keep for similarity with original code)
                (batch, -1) + weight.shape[1:]
            )
            # switch places between in_ch and out_ch
            # put batch as part of in_ch
            weight = weight.transpose(1, 2).reshape(
                (batch * in_channel, self.out_channel_expanded) + weight.shape[3:]
            )

            weight = self.skeleton_traits.reshape_weight_before_transposed_conv(weight, batch,
                                                                                in_channel, self.out_channel)

            out = self.skeleton_traits.transposed_conv_func(input, weight, padding=self.updown_pad,
                                                            stride=self.skeleton_traits.updown_stride, groups=batch)

            # reshape s.t. batch is in a separate dim. works for both conv2 and conv3
            out = out.view((batch, self.out_channel) + out.shape[-2:])

            # the blur is not making a significant change in size,
            # just dropping the redundant rows/columns (e.g., from (3,9) to (2,8))
            out = self.skeleton_traits.blur(self.blur, out)

        else:  # keep dims
            # ==>  batch, in_channel,[1,] in_height, in_width
            input = self.skeleton_traits.reshape_input_before_conv(input, batch, width)

            input = input.view((1, batch * input.shape[1]) + input.shape[2:])  # ==> 1, batch*in_ch, ...
            out = self.skeleton_traits.conv_func(input, weight, padding=self.fixed_dim_pad, groups=batch)
            out = self.skeleton_traits.reshape_output_after_conv(out)  # ==> 1, batch*out_ch, height, width

            out = out.view(batch, self.out_channel, out.shape[2], out.shape[-1])  # ==> batch, out_ch, height, width

        if self.demod_obj == 'data':
            instance_std = out.std((2, 3), keepdim=True)  # batch x ch
            out = out / instance_std

        return out


class ConstantInput(nn.Module):
    def __init__(self, channel, size=(1, 4)):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size[0], size[1]))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        skeleton_traits=None
    ):
        super().__init__()

        self.conv = ModulatedConv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            skeleton_traits=skeleton_traits
        )

        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style):
        out = self.conv(input, style)
        out = self.activate(out)

        return out


class ToXYZ(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1),
                 skeleton_traits=None, skip_pooling_list=None, motion_statics: StaticData=None):
        super().__init__()

        self.skel_aware = skeleton_traits.skeleton_aware()

        # this is true when applying the 'skip' branch, for upsampling the 'skip' features from previous hierarchy level
        if upsample:

            # upsample and unpool are destined for 'skip', which possesses the shape of prev hierarchy level
            self.upsample = skeleton_traits.upsample(blur_kernel)
            if self.skel_aware:
                self.skeleton_unpool = SkeletonUnpool(pooling_list=skip_pooling_list,
                                                      output_joints_num=skeleton_traits.larger_n_joints)

        # conv is destined for the input, which is already upsampled
        self.conv = ModulatedConv(in_channel, motion_statics.n_channels, 1, style_dim,
                                  demodulate=False, skeleton_traits=skeleton_traits)
        self.bias = nn.Parameter(torch.zeros(1, motion_statics.n_channels, 1, 1))


    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            if self.skel_aware:
                skip = self.skeleton_unpool(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        n_mlp,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        traits_class=None,
        motion_statics=None,
        n_inplace_conv=1
    ):
        super().__init__()

        self.traits_class = traits_class
        n_joints = traits_class.n_joints(motion_statics)
        self.n_channels = traits_class.n_channels(motion_statics)
        self.n_frames = traits_class.n_frames(motion_statics)

        # unlike stylegan2 for images, here target size is a const.
        # not used but kept here for similarity with original code
        self.size = (n_joints[-1], self.n_frames[-1])

        self.style_dim = style_dim

        layers = [PixelNorm()]

        # mapping (style) network and constant
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)  # MLP layer f: Z -> W.
        self.input = ConstantInput(self.n_channels[0], size=(n_joints[0], self.n_frames[0]))
        # end mapping network and constant

        layer0 = motion_statics.hierarchical_keep_dim_layer(layer=0)
        skeleton_traits1 = traits_class(layer0)
        self.conv1 = StyledConv(
            self.n_channels[0], self.n_channels[0], 3, style_dim,
            blur_kernel=blur_kernel, skeleton_traits=skeleton_traits1
        )
        self.to_xyz1 = ToXYZ(self.n_channels[0], style_dim, upsample=False,
                             skeleton_traits=skeleton_traits1, motion_statics=motion_statics)

        if traits_class.is_pool():
            n_inplace_conv -= 1  # the pooling block already contains one inplace convolution
        self.n_inplace_convs_in_hierarchy = n_inplace_conv
        self.n_convs_in_hierarchy = 1 + n_inplace_conv
        self.n_hierarchy_levels = len(n_joints)
        self.n_total_conv_layers = (self.n_hierarchy_levels - 1) * self.n_convs_in_hierarchy + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_xyzs = nn.ModuleList()

        in_channel = self.n_channels[0]

        for i in range(1, len(self.n_channels)):
            out_channel = self.n_channels[i]
            cur_parents = motion_statics.parents_list[i]

            layer_i = motion_statics.hierarchical_upsample_layer(layer=i)
            skeleton_traits_upsample = traits_class(layer_i)
            # upsample
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    skeleton_traits=skeleton_traits_upsample,
                )
            )

            layer_i = motion_statics.hierarchical_keep_dim_layer(layer=i)
            skeleton_traits_keep_dims = traits_class(layer_i)
            # keep dims
            for _ in range(n_inplace_conv):
                self.convs.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                        skeleton_traits=skeleton_traits_keep_dims
                    )
                )

            self.to_xyzs.append(ToXYZ(out_channel, style_dim, skeleton_traits=skeleton_traits_keep_dims,
                                      skip_pooling_list=motion_statics.skeletal_pooling_dist_1[i - 1], motion_statics =motion_statics ))

            in_channel = out_channel

        # number of style codes to be injected (w height)
        # (len(n_joints)-1) * 3  + 2 # 1st level gets latent[0],
        # next each level i gets latent[i*2-1,i*2], motion (i.e. last skip) gets i*2+1
        self.n_latent = self.n_total_conv_layers + 1

        # keep names of parameters that should have required_grad=False
        self.non_grad_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                self.non_grad_params.append(name)

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        return_sub_motions=False,
    ):
        if not input_is_latent:
            # forward the noise through the style pipeline to obtain W
            styles = [self.style(s) for s in styles]

        if truncation < 1: # apply truncation trick if needed. I.e., use a portion of W that is closer to the center
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        # repeat W n_latent times (if we use 2 style inputs repeat one inject index times and the other [n_latent - inject_index times])
        # call it latent
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            elif styles[0].shape[1] == 1:
                latent = styles[0].expand(styles[0].shape[0], self.n_latent_needed, styles[0].shape[2])
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            if isinstance(styles, torch.Tensor) and len(styles.shape) == 3 and styles.shape[1] == 1:
                # W space.
                latent = styles.expand(styles.shape[0], self.n_latent, styles.shape[2])
            else:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)  # duplicate self.constant batch_size times. latent is used only to know batch size. out.shape is [16, 256, 1, 4]j or [16, 64, 6, 16]e
        out = self.conv1(out, latent[:, 0])  # out.shape is [16, 256, 1, 4]j / [16, 64, 6, 16]e (keep dims)

        motion = list()
        skip = self.to_xyz1(out, latent[:, 1]) # skip.shape is [16, 3, 1, 4]j [16, 4, 6, 16]e
        if return_sub_motions:  # return all motions created by all pyramid layers
            motion.append(skip)

        i = 1
        for to_xyz in self.to_xyzs:
            out = self.convs[i-1](out, latent[:, i])      # upsample [16, 64, 12, 32]e, [16, 32, 19, 64]e
            for j in range(i, i + self.n_inplace_convs_in_hierarchy):
                out = self.convs[j](out, latent[:, j])  # keep dims
            skip = to_xyz(out, latent[:, i + self.n_inplace_convs_in_hierarchy + 1], skip)        # [16, 4, 11, 32]e, [16, 4, 18, 64]e
            if return_sub_motions: # return all motions created by all pyramid layers
                motion.append(skip)

            i += self.n_convs_in_hierarchy

        if not return_sub_motions: # return final motion only
            motion = skip

        if return_latents:
            return motion, latent, inject_index

        else:
            return motion, None, None  # [16, 4, 20, 64]e

# encapsulate optional downsample, convolution and non linearity
# used only by the DISCRIMINATOR
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        skeleton_traits=None,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            pad = skeleton_traits.upfirdn_pad(pad0, pad1)

            if (skeleton_traits.need_blur):
                layers.append(Blur(blur_kernel, pad, skeleton_traits=skeleton_traits)) # not decreasing size, but fixing #frame, e.g., from 64 frames to 65 so conv will work with stride 2.
            stride = skeleton_traits.updown_stride
            padding = skeleton_traits.updown_pad(kernel_size)

        else:  # keep dims
            stride = 1
            padding = skeleton_traits.fixed_dim_pad(kernel_size)

        layers.append(
            EqualConv(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
                skeleton_traits=skeleton_traits
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))  # scaled leaky relu over (layers+bias)
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, skeleton_traits_for_kernel_3,
                 skeleton_traits_for_kernel_1, skeleton_traits_keep_dims, n_inplace_conv=1):
        super().__init__()

        self.n_inplace_conv = n_inplace_conv

        convs = []
        for _ in range(n_inplace_conv):
            convs.append(ConvLayer(in_channel, in_channel, 3, skeleton_traits=skeleton_traits_keep_dims))  # keep dims - inplace conv
        convs.append(ConvLayer(in_channel, out_channel, 3, downsample=True, skeleton_traits=skeleton_traits_for_kernel_3))  # downsample, kernel=3
        self.convs = nn.Sequential(*convs)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, skeleton_traits=skeleton_traits_for_kernel_1 # downsample, kernel=1
        )

    def forward(self, input):
        out = self.convs(input)

        skip = self.skip(input) # downscale, kernel=1
        out = (out + skip) / math.sqrt(2) # division by sqrt(2) for residual blocks is explained in footnote (3) in the stylegan2 paper

        return out


class Discriminator(nn.Module):
    def __init__(self, blur_kernel=[1, 3, 3, 1], traits_class=None, motion_statics =None, reconstruct_latent=False, latent_dim=None, latent_rec_idx=None, n_inplace_conv=1,
                 n_latent_predict=0):
        super().__init__()

        if traits_class.is_pool():
            n_inplace_conv -= 1  # the pooling block already contains one inplace convolution
        n_joints = traits_class.n_joints(motion_statics )
        self.n_channels = traits_class.n_channels(motion_statics )
        self.n_frames = traits_class.n_frames(motion_statics )
        self.n_levels = traits_class.n_levels(motion_statics )

        layer_i = motion_statics .hierarchical_keep_dim_layer(layer=-1)
        skeleton_traits = traits_class(layer_i)

        convs = [ConvLayer(motion_statics .n_channels, self.n_channels[-1], 1, skeleton_traits=skeleton_traits)] # channel-wise expansion. keep dims. kernel=1

        in_channel = self.n_channels[-1]

        for i in range(self.n_levels-1, 0, -1):
            out_channel = self.n_channels[i-1]

            upsample_layer_1 = motion_statics .hierarchical_upsample_layer(layer=i)
            upsample_layer_0 = motion_statics .hierarchical_upsample_layer(layer=i, pooling_dist=0)
            keep_layer = motion_statics .hierarchical_keep_dim_layer(layer=i)

            skeleton_traits_for_kernel_3 = traits_class(upsample_layer_1)
            skeleton_traits_for_kernel_1 = traits_class(upsample_layer_0)
            skeleton_traits_keep_dims = traits_class(keep_layer)

            convs.append(ResBlock(in_channel, out_channel, skeleton_traits_for_kernel_3,
                                  skeleton_traits_for_kernel_1, skeleton_traits_keep_dims, n_inplace_conv))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        layer_i = motion_statics .hierarchical_keep_dim_layer(layer=0)
        skeleton_traits = traits_class(layer_i)

        self.final_conv = ConvLayer(in_channel + 1, self.n_channels[0], 3, skeleton_traits=skeleton_traits) # channels 257-->256, keep dims, kernel=3
        self.final_linear = nn.Sequential(
            EqualLinear(self.n_channels[0] * n_joints[0] * self.n_frames[0], self.n_channels[0], activation='fused_lrelu'),
            EqualLinear(self.n_channels[0], 1),
        )
        self.latent_rec_idx = latent_rec_idx

        self.n_latent_predict = n_latent_predict
        if n_latent_predict > 0:
            i = max(0, self.n_levels - latent_rec_idx)
            self.latent_predictor = LatentPrediction(self.n_channels[i] * n_joints[i] * self.n_frames[i], latent_dim,
                                                     n_latent_predict)
        else:
            self.latent_predictor = None

        # keep names of parameters that should have requires_grad=False
        self.non_grad_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                self.non_grad_params.append(name)


    def forward(self, input):
        # input dims: (samples, channels, entities, frames)
        # out = self.convs(input)
        for i, module in enumerate(self.convs):
            if i == self.latent_rec_idx:
                latent_base = input
            input = module(input)

        if self.latent_rec_idx == len(self.convs):
            latent_base = input
        out = input
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)   # mean of std across channels, joints, frames
        stddev = stddev.repeat(group, 1, height, width)             # repeat across batch, joints, frames
        out = torch.cat([out, stddev], 1)                           # stddev added as an additional channel

        out = self.final_conv(out)                                  # fuse the additional stddev channel with existing ones

        out = out.view(batch, -1)
        if self.latent_rec_idx == len(self.convs) + 1 or self.latent_rec_idx == -1:
            latent_base = out
        if self.latent_predictor is not None:
            rec_latent = self.latent_predictor(latent_base)
        else:
            rec_latent = None

        if not self.latent_predictor:
            out = self.final_linear(out)
        else:
            out = None

        return out, rec_latent


def keep_skeletal_dims(n_joints):
    return {joint_idx: [joint_idx] for joint_idx in range(n_joints)}