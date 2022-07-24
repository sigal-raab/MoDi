import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonPool(nn.Module):
    def __init__(self, pooling_list, pooling_mode, input_joints_num):
        super(SkeletonPool, self).__init__()

        if pooling_mode != 'mean':
            raise Exception('Unimplemented pooling mode in matrix_implementation')

        self.output_joints_num = len(pooling_list)
        self.input_joints_num = input_joints_num
        self.pooling_mode = pooling_mode

        self.weight = torch.zeros(self.output_joints_num, self.input_joints_num)

        for i, pair in pooling_list.items():
            for j in pair:
                self.weight[i , j] = 1.0 / len(pair)

        self.weight = nn.Parameter(self.weight.to('cuda'), requires_grad=False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input)


class SkeletonUnpool(nn.Module):
    def __init__(self, pooling_list, output_joints_num):
        super(SkeletonUnpool, self).__init__()
        self.pooling_list = pooling_list
        self.input_joints_num = len(pooling_list)
        self.output_joints_num = output_joints_num

        self.weight = torch.zeros(self.output_joints_num, self.input_joints_num)

        for i, affecting_joints in self.pooling_list.items():
            for j in affecting_joints:
                self.weight[j, i] = 1

        # if an output joint is affected by more than one input joint, it takes the average of all contributors
        self.weight = F.normalize(self.weight, p=1)

        self.weight = nn.Parameter(self.weight.to('cuda'), requires_grad=False)

    def forward(self, input: torch.Tensor):
        #input size: batch x ch x j_in x t
        # weights size: j_out x j_in
        # matmul size: batch x ch x j_out x t
        return torch.matmul(self.weight, input)





