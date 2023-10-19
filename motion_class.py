import yaml
import functools

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Motion import BVH
from Motion.Animation import Animation
from utils.foot import get_foot_location
from Motion.Quaternions import Quaternions
from Motion.AnimationStructure import children_list, get_sorted_order
from utils.data import expand_topology_edges, Joint


DEFAULT_CONFIG = 'default'


class EdgePoint(tuple):
    def __new__(cls, a, b):
        return super(EdgePoint, cls).__new__(cls, [a, b])

    def __repr__(self):
        return f'Edge{super(EdgePoint, self).__repr__()}'


class StaticConfig:
    CONFIG_YAML_FILE = 'utils/config.yaml'

    def __getitem__(self, item):
        assert item in self.default_config
        return self.config.get(item, self.default_config[item])

    def __init__(self, character_name: str):
        config = self.load_config(self.CONFIG_YAML_FILE)
        
        self.character_name = character_name
        self.default_config = config[DEFAULT_CONFIG]
        self.config = config[character_name] if character_name in config else config[DEFAULT_CONFIG]

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

        return config


class StaticMotionOneHierarchyLevel:
    """Representing a static layer in the down sampling process including parents and feet indexes.

    Attributes:
        parents: List of parents  at specific hierarchy level - representing the skeleton.
        pooling_list: Dictionary from every joint in that hierarchy level to the skeleton in the next one.
        use_global_position: flag whether the model is using global position.
        feet_indices list on feet indices in case foot_contact is on o.w empty list.
    """
    def __init__(self, parents: [int], pooling_list: {int: [int]},  use_global_position: bool, feet_indices: [str]):
        self.parents = parents
        self.pooling_list = pooling_list
        self.use_global_position = use_global_position
        self.feet_indices = feet_indices

    @classmethod
    def keep_dim_layer(cls, parents: [int], *args, **kwargs):
        layer = cls(parents, None, *args, **kwargs)
        layer.pooling_list = {joint_idx: [joint_idx] for joint_idx in range(layer.edges_number)}

        return layer

    @property
    def foot_contact(self):
        return self.feet_indices is not None

    @property
    def edges_number(self):
        return len(self.parents) + self.use_global_position + len(self.feet_indices)

    @property
    def edges_number_after_pooling(self):
        return len(self.pooling_list)

    def fake_parents(self, parents=None):
        # In case debug is needed to be compared to older method.
        if not parents:
            parents = self.parents
        return parents + ([-2] if self.use_global_position else []) + ([(-3, index) for index in self.feet_indices])


class StaticData:
    def __init__(self, parents: [int], offsets: np.array, names: [str], character_name: str,
                 n_channels=4, enable_global_position=False, enable_foot_contact=False,
                 rotation_representation='quaternion'):
        self._offsets = offsets.copy() if offsets is not None else None
        self.names = names.copy() if names is not None else None
        self.config = StaticConfig(character_name)

        self.parents_list, self.skeletal_pooling_dist_1_edges = self.calculate_all_pooling_levels(parents)
        self.skeletal_pooling_dist_1 = [{edge[1]: [e[1] for e in pooling[edge]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        self.skeletal_pooling_dist_0 = [{edge[1]: [pooling[edge][-1][1]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        # Configurations
        self.__n_channels = n_channels

        self.enable_global_position = enable_global_position
        self.enable_foot_contact = enable_foot_contact

        if enable_global_position:
            self._enable_global_position()
        if enable_foot_contact:
            self._enable_foot_contact()
        if rotation_representation == 'repr6d':
            self._enable_repr6d()

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str, *args, **kwargs):
        animation, names, frametime = BVH.load(bvf_filepath)
        return cls(animation.parents, animation.offsets, names, *args, **kwargs)

    @classmethod
    def init_from_motion(cls, motion, **kwargs):
        offsets = np.concatenate([motion['offset_root'][np.newaxis, :], motion['offsets_no_root']])
        return cls(motion['parents_with_root'], offsets, motion['names_with_root'], **kwargs)

    @classmethod
    def init_joint_static(cls, joint: Joint, **kwargs):
        motion_statics = cls(joint.parents_list[-1], offsets=None, names=joint.parents_list[-1], n_channels=3,
                     enable_foot_contact=False, rotation_representation=False, **kwargs)
        motion_statics.parents_list = joint.parents_list
        motion_statics.skeletal_pooling_dist_1 = joint.skeletal_pooling_dist_1
        motion_statics.skeletal_pooling_dist_0 = joint.skeletal_pooling_dist_0

        return motion_statics

    @property
    def parents(self):
        return self.parents_list[-1][:len(self.names)]

    @property
    def entire_motion(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def offsets(self) -> np.ndarray:
        return self._offsets.copy() if self._offsets is not None else None

    @staticmethod
    def edge_list(parents: [int]) -> [EdgePoint]:
        return [EdgePoint(dst, src + 1) for src, dst in enumerate(parents[1:])]

    @property
    def n_channels(self) -> int:
        return self.__n_channels

    def _enable_repr6d(self):
        self.__n_channels = 6

    def _enable_marker4(self):
        self.__n_channels = 12
    # @n_channels.setter
    # def n_channels(self, val: int) -> None:
    #     self.__n_channels = val

    @property
    def character_name(self):
        return self.config.character_name

    @property
    def n_edges(self):
        return [len(parents) for parents in self.parents_list]

    def save_to_bvh(self, out_filepath: str) -> None:
        raise NotImplementedError

    def _enable_global_position(self):
        """
        add a special entity that would be the global position.
        The entity is appended to the edges list.
        No need to really add it in edges_list and all the other structures that are based on tuples. We add it only
        to the structures that are based on indices.
        Its neighboring edges are the same as the neighbors of root """
        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage in pooling_list:
                n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                n_large_stage = max(val for edge in pooling_hierarchical_stage.values() for val in edge) + 1
                pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

    @property
    def foot_names(self):
        return self.config['feet_names']

    @property
    @functools.lru_cache()
    def foot_indexes(self, include_toes=True):
        """Run overs pooling list and calculate foot location at each level"""
        # feet_names = [LEFT_FOOT_NAME, LEFT_TOE, RIGHT_FOOT_NAME, RIGHT_TOE] if include_toes else [LEFT_FOOT_NAME,
        #                                                                                           RIGHT_FOOT_NAME]

        indexes = [i for i, name in enumerate(self.names) if name in self.foot_names]
        all_foot_indexes = [indexes]
        for pooling in self.skeletal_pooling_dist_1[::-1]:
            all_foot_indexes += [[k for k in pooling if any(foot in pooling[k] for foot in all_foot_indexes[-1])]]

        return all_foot_indexes[::-1]

    @property
    def foot_number(self):
        return len(self.foot_indexes[-1])

    def _enable_foot_contact(self):
        """ add special entities that would be the foot contact labels.
        The entities are appended to the edges list.
        No need to really add them in edges_list and all the other structures that are based on tuples. We add them only
        to the structures that are based on indices.
        Their neighboring edges are the same as the neighbors of the feet """
        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage, foot_indexes in zip(pooling_list, self.foot_indexes):
                for _ in foot_indexes:
                    n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                    n_large_stage = max(val for edge in pooling_hierarchical_stage.values() for val in edge) + 1
                    pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

    def hierarchical_upsample_layer(self, layer: int, pooling_dist=1) -> StaticMotionOneHierarchyLevel:
        assert pooling_dist in [0, 1]
        skeletal_pooling_dist = self.skeletal_pooling_dist_1 if pooling_dist == 1 else self.skeletal_pooling_dist_0

        return StaticMotionOneHierarchyLevel(self.parents_list[layer],
                                             skeletal_pooling_dist[layer - 1],
                                             self.enable_global_position,
                                             self.foot_indexes[layer] if self.enable_foot_contact else [])

    def hierarchical_keep_dim_layer(self, layer: int) -> StaticMotionOneHierarchyLevel:
        return StaticMotionOneHierarchyLevel.keep_dim_layer(self.parents_list[layer],
                                                            self.enable_global_position,
                                                            self.foot_indexes[layer] if self.enable_foot_contact else [])
    
    def number_of_joints_in_hierarchical_levels(self) -> [int]:
        feet_lengths = [len(feet) for feet in self.foot_indexes] if self.enable_foot_contact else [0] * len(self.parents_list)
        return [len(parents) + self.enable_global_position + feet_length
                 for parents, feet_length in zip(self.parents_list, feet_lengths)]

    @staticmethod
    def _topology_degree(parents: [int]):
        joints_degree = [0] * len(parents)

        for joint in parents[1:]:
            joints_degree[joint] += 1

        return joints_degree

    @staticmethod
    def _find_seq(index: int, joints_degree: [int], parents: [int]) -> [[int]]:
        """Recursive search to find a list of all straight sequences of a skeleton."""
        if joints_degree[index] == 0:
            return [[index]]

        all_sequences = []
        if joints_degree[index] > 1 and index != 0:
            all_sequences = [[index]]

        children_list = [dst for dst, src in enumerate(parents) if src == index]

        for dst in children_list:
            sequence = StaticData._find_seq(dst, joints_degree, parents)
            sequence[0] = [index] + sequence[0]
            all_sequences += sequence

        return all_sequences

    @staticmethod
    def _find_leaves(index: int, joints_degree: [int], parents: [int]) -> [[int]]:
        """Recursive search to find a list of all leaves and their connected joint in a skeleton rest position"""
        if joints_degree[index] == 0:
            return []

        all_leaves_pool = []
        connected_leaves = []

        children_list = [dst for dst, src in enumerate(parents) if src == index]

        for dst in children_list:
            leaves = StaticData._find_leaves(dst, joints_degree, parents)
            if leaves:
                all_leaves_pool += leaves
            else:
                connected_leaves += [dst]

        if connected_leaves:
            all_leaves_pool += [[index] + connected_leaves]

        return all_leaves_pool

    @staticmethod
    def _edges_from_joints(joints: [int]):
        return [(src, dst) for src, dst in zip(joints[:-1], joints[1:])]

    @staticmethod
    def _pooling_for_edges_list(edges: [EdgePoint]) -> list:
        """Return a list sublist of edges of length 2."""
        pooling_groups = [edges[i:i + 2] for i in range(0, len(edges), 2)]
        if len(pooling_groups) > 1 and len(pooling_groups[-1]) == 1:  # If we have an odd numbers of edges pull 3 of them in once.
            pooling_groups[-2] += pooling_groups[-1]
            pooling_groups = pooling_groups[:-1]

        return pooling_groups

    @staticmethod
    def flatten_dict(values):
        return {k: sublist[k] for sublist in values for k in sublist}

    @staticmethod
    def _calculate_degree1_pooling(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        """Pooling for complex skeleton by trimming long sequences into smaller ones."""
        all_sequences = StaticData._find_seq(0, degree, parents)
        edges_sequences = [StaticData._edges_from_joints(seq) for seq in all_sequences]

        pooling = [{(edge[0][0], edge[-1][-1]): edge for edge in StaticData._pooling_for_edges_list(edges)} for edges in
                   edges_sequences]
        pooling = StaticData.flatten_dict(pooling)

        return pooling

    @staticmethod
    def _calculate_leaves_pooling(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        # all_leaves = StaticData._find_leaves(0, degree, parents)
        all_sequences = StaticData._find_seq(0, degree, parents)
        edges_sequences = [StaticData._edges_from_joints(seq) for seq in all_sequences]

        all_joints = [joint for joint, d in enumerate(degree) if d > 0]
        pooling = {}

        for joint in all_joints:
            pooling[joint] = [edge[0] for edge in edges_sequences if edge[0][0] == joint]

        return {pooling[k][0]: pooling[k] for k in pooling}

    @staticmethod
    def _calculate_pooling_for_level(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        if any(d == 1 for d in degree):
            return StaticData._calculate_degree1_pooling(parents, degree)
        else:
            return StaticData._calculate_leaves_pooling(parents, degree)

    @staticmethod
    def _normalise_joints(pooling: {EdgePoint: [EdgePoint]}) -> {EdgePoint: [EdgePoint]}:
        max_joint = 0
        joint_to_new_joint: {int: int} = {-1: -1, 0: 0}
        new_edges = {}

        for edge in sorted(pooling, key=lambda x: x[1]):
            if edge[1] > max_joint:
                max_joint += 1
                joint_to_new_joint[edge[1]] = max_joint

            new_joint = tuple(joint_to_new_joint[e] for e in edge)
            new_edges[new_joint] = pooling[edge]

        return new_edges

    @staticmethod
    def _edges_to_parents(edges: [EdgePoint]):
        return [edge[0] for edge in edges]

    def calculate_all_pooling_levels(self, parents0):
        all_parents = [list(parents0)]
        all_poolings = []
        degree = StaticData._topology_degree(all_parents[-1])

        while len(all_parents[-1]) > 2:
            pooling = self._calculate_pooling_for_level(all_parents[-1], degree)
            pooling[(-1, 0)] = [(-1, 0)]

            normalised_pooling = self._normalise_joints(pooling)
            normalised_parents = self._edges_to_parents(normalised_pooling.keys())

            all_parents += [normalised_parents]
            all_poolings += [normalised_pooling]

            degree = StaticData._topology_degree(all_parents[-1])

        # all_parents += [[-1, 0, 1], [-1]]
        # all_poolings += [{(-1, 0): [(-1, 0)], (0, 1): [(0, 1), (0, 5), (0, 6)], (1, 2): [(1, 2), (1, 3), (1, 4)]},
        #                  {(-1, 0): [(-1, 0), (0, 1), (1, 2)]}]

        return all_parents[::-1], all_poolings[::-1]

    def plot(self, parents, show=True):
        graph = nx.Graph()
        graph.add_edges_from(self.edge_list(parents))
        nx.draw_networkx(graph)
        if show:
            plt.show()


class DynamicData:
    def __init__(self, motion: torch.tensor, motion_statics: StaticData, use_velocity=False):
        self.motion = motion.clone()  # Shape is B  x K x J x T = batch x channels x joints x frames
        self.motion_statics = motion_statics

        # self.assert_shape_is_right()

        self.use_velocity = use_velocity

    def assert_shape_is_right(self):
        foot_contact_joints = self.motion_statics.foot_number if self.motion_statics.enable_foot_contact else 0
        global_position_joint = 1 if self.motion_statics.enable_global_position else 0

        assert self.motion.shape[-3] == self.motion_statics.n_channels
        assert self.motion.shape[-2] == len(self.motion_statics.parents) + global_position_joint + foot_contact_joints

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str, character_name: str,
                      enable_global_position=False,
                      enable_foot_contact=False,
                      use_velocity=False,
                      rotation_representation='quaternion'):

        animation, names, frametime = BVH.load(bvf_filepath)

        motion_statics = StaticData(animation.parents, animation.offsets, names, character_name,
                            enable_global_position=enable_global_position,
                            enable_foot_contact=enable_foot_contact,
                            rotation_representation=rotation_representation)

        return cls(animation.rotations.qs, motion_statics , use_velocity=use_velocity)

    def sub_motion(self, motion):
        return self.__class__(motion, self.motion_statics , use_velocity=self.use_velocity)

    def __iter__(self):
        if self.motion.ndim == 4:
            return (self.sub_motion(motion) for motion in self.motion).__iter__()
        elif self.motion.ndim == 3:
            return [self.sub_motion(self.motion)].__iter__()

    def __getitem__(self, slice_val):
        return self.sub_motion(self.motion[slice_val])

    @property
    def shape(self):
        return self.motion.shape

    @property
    def n_frames(self):
        return self.motion.shape[-1]

    @property
    def n_channels(self):
        return self.motion.shape[-3]

    @property
    def n_joints(self):
        return len(self.motion_statics .names)

    @property
    def edge_rotations(self) -> torch.tensor:
        return self.motion[..., :self.n_joints, :].detach().cpu()
        # Return only joints representing motion, maybe having a batch dim

    def foot_contact(self) -> torch.tensor:
        return [[{foot: motion[0, self.n_joints + 1 + idx, frame].numpy().item()
                for idx, foot in enumerate(self.motion_statics .foot_names)} for frame in range(self.n_frames)]
                for motion in self.motion]

    @property
    def root_location(self) -> torch.tensor:
        location = self.motion[..., :3, self.n_joints, :].detach().cpu()  # drop the 4th item in the position tensor
        location = np.cumsum(location, axis=1) if self.use_velocity else location

        return location  # K x T

    def un_normalise(self, mean: torch.tensor, std: torch.tensor):
        return self.sub_motion(self.motion * std + mean)

    def sample_frames(self, frames_indexes: [int]):
        return self.sub_motion(self.motion[..., frames_indexes])

    def basic_anim(self):
        offsets = self.motion_statics.offsets

        positions = np.repeat(offsets[np.newaxis], self.n_frames, axis=0)
        positions[:, 0] = self.root_location.transpose(0, 1)

        orients = Quaternions.id(self.n_joints)
        rotations = Quaternions(self.edge_rotations.permute(2, 1, 0).numpy())

        if rotations.shape[-1] == 6:  # repr6d
            from Motion.transforms import repr6d2quat
            rotations = repr6d2quat(rotations)

        anim_edges = Animation(rotations, positions, orients, offsets, self.motion_statics .parents)

        return anim_edges

    def move_rotation_values_to_parents(self, anim_exp):
        children_all_joints = children_list(anim_exp.parents)
        for idx, children_one_joint in enumerate(children_all_joints[1:]):
            parent_idx = idx + 1
            if len(children_one_joint) > 0:  # not leaf
                assert len(children_one_joint) == 1 or (anim_exp.offsets[children_one_joint] == np.zeros(3)).all() and (
                        anim_exp.rotations[:, children_one_joint] == Quaternions.id((self.n_frames, 1))).all()
                anim_exp.rotations[:, parent_idx] = anim_exp.rotations[:, children_one_joint[0]]
            else:
                anim_exp.rotations[:, parent_idx] = Quaternions.id((self.n_frames))

        return anim_exp

    def to_anim(self):
        anim_edges = self.basic_anim()

        sorted_order = get_sorted_order(anim_edges.parents)
        anim_edges_sorted = anim_edges[:, sorted_order]
        names_sorted = self.motion_statics.names[sorted_order]

        anim_exp, _, names_exp, _ = expand_topology_edges(anim_edges_sorted, names=names_sorted, nearest_joint_ratio=1)
        anim_exp = self.move_rotation_values_to_parents(anim_exp)

        return anim_exp, names_exp


def plot(name: str):
    """Decorator for any plot method to create a new figure, title and save it."""
    def decorator(plot_method: callable):
        def wrapper(self, *args, **kwargs):
            plt.figure()
            plot_method(self, *args, **kwargs)
            plt.title(name)
            plt.savefig(f'fc/train_{name.replace(" ", "_")}')
        return wrapper
    return decorator


class DebugDynamic(DynamicData):
    @property
    def foot_location(self):
        label_idx = self.n_joints + self.motion_statics .enable_global_position
        location, _, _ = get_foot_location(self.motion[:, :, :label_idx], self.motion_statics ,
                                           use_global_position=self.motion_statics .enable_global_position,
                                           use_velocity=self.use_velocity)

        return location

    @property
    def foot_velocity(self):
        return (self.foot_location[:, 1:] - self.foot_location[:, :-1]).pow(2).sum(axis=-1).sqrt()

    @property
    def predicted_foot_contact(self):
        label_idx = self.n_joints + self.motion_statics .enable_global_position
        predicted_foot_contact = self.motion[..., 0, label_idx:, :]
        return torch.sigmoid((predicted_foot_contact - 0.5) * 2 * 6)

    @property
    def foot_contact_loss(self):
        return self.predicted_foot_contact[..., 1:] * self.foot_velocity

    def foot_movement_index(self) -> float:
        """ calculate the amount of foot movement."""
        foot_location = self.motion[..., self.motion_statics .foot_indexes[-1], :].clone()  # B x K x feet x T
        foot_velocity = ((foot_location[..., 1:] - foot_location[..., :-1]) ** 2).sum(axis=1)  # B x feet x T

        return foot_velocity.sum(axis=1).sum(axis=1)

    @plot('foot location')
    def plot_foot_location(self, index) -> None:
        my_foot = self.foot_location[index, ..., 1].detach().cpu().numpy()
        plt.plot(my_foot)

    @plot('foot velocity')
    def plot_foot_velocity(self, index) -> None:
        my_vel = self.foot_velocity[index].detach().cpu().numpy()
        plt.plot(my_vel)

    @plot('foot contact')
    def plot_foot_contact_labels(self, index) -> None:
        plt.plot(self.predicted_foot_contact[index].transpose(0, 1).detach().cpu().numpy())

    @plot('loss')
    def plot_foot_contact_loss(self, index) -> None:
        plt.plot(self.foot_contact_loss)

    def plot_all_4(self, index):
        self.plot_foot_location(index)
        self.plot_foot_velocity(index)
        self.plot_foot_contact_labels(index)
        self.plot_foot_contact_loss(index)
