import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
os.makedirs(os.path.join(os.path.expanduser('~'),'tmp', 'stylegan_lock'), exist_ok=True)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
    build_directory=os.path.join(os.path.expanduser('~'),'tmp', 'stylegan_lock'),
    # with_cuda=False
)


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    """
    x = input+bias
    if x > 0: out = (x+bias)
    else      out = (x+bias) * negative_slope
    out *= scale
    multiplication by scale is to get the std back to 1 (like in Kaiming He initialization)
    """
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        # import math
        # ratio = 0.5 + 0.5 * self.negative_slope
        # mult = 1/ratio
        # scale1 = math.sqrt(mult)
        # scale2 = nn.init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(1/self.negative_slope)) # sqrt(2/(1+param**2))
        # scale3 = nn.init.calculate_gain(nonlinearity='leaky_relu', param=self.negative_slope)
        # scale4 = nn.init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(self.negative_slope)) # equivalent to scale1
        #
        # print('FusedLeakyReLU')
        # print([input.std(), fused_leaky_relu(input, self.bias, self.negative_slope, scale1).std()])
        # print([input.std(), fused_leaky_relu(input, self.bias, self.negative_slope, scale2).std()])
        # print([input.std(), fused_leaky_relu(input, self.bias, self.negative_slope, scale3).std()])
        # print([input.std(), fused_leaky_relu(input, self.bias, self.negative_slope, scale4).std()])
        # print([input.std(), fused_leaky_relu(input, self.bias, self.negative_slope, math.sqrt(2)).std()])
        # print([input.std(), fused_leaky_relu(input, self.bias, self.negative_slope, 1).std()])
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == "cpu":
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            )
            * scale
        )

    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
