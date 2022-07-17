##############################
#
# based on https://github.com/PeizhuoLi/ganimator/blob/main/models/kinematics.py
#
##############################

import torch
from Motion.transforms import quat2mat, repr6d2mat, euler2mat


class ForwardKinematicsJoint:
    def __init__(self, parents, offset):
        self.parents = parents
        self.offset = offset

    '''
        rotation should have shape batch_size * Time * Joint_num * (3/4)
        position should have shape batch_size * Time * 3
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset=None,
                world=True):

        if rotation.shape[-1] == 6:
            transform = repr6d2mat(rotation)
        elif rotation.shape[-1] == 4:
            norm = torch.norm(rotation, dim=-1, keepdim=True)
            rotation = rotation / norm
            transform = quat2mat(rotation)
        elif rotation.shape[-1] == 3:
            transform = euler2mat(rotation)
        else:
            raise Exception('Only accept quaternion rotation input')
        result = torch.empty(transform.shape[:-2] + (3,), device=position.device)

        if offset is None:
            offset = self.offset
        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            if world: result[..., i, :] += result[..., pi, :]
        return result


    def forward_edge_rot(self, rotation: torch.Tensor, position: torch.Tensor, offset=None,
                world=True):
        """ A slightly different fk, because we keep an edge's rotation in itself and not in its parent """

        if rotation.shape[-1] == 6:
            transform = repr6d2mat(rotation)
        elif rotation.shape[-1] == 4:
            norm = torch.norm(rotation, dim=-1, keepdim=True)
            rotation = rotation / norm
            transform = quat2mat(rotation)
        elif rotation.shape[-1] == 3:
            transform = euler2mat(rotation)
        else:
            raise Exception('Only accept quaternion rotation input')
        result = torch.empty(transform.shape[:-2] + (3,), device=position.device)

        if offset is None:
            offset = self.offset
        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            result[..., i, :] = torch.matmul(transform[..., i, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result
