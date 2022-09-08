import os
import os.path as osp
import sys
import shutil
from glob import glob
import numpy as np
import re
import csv
import pandas as pd
from sklearn import svm

from Motion import BVH
from Motion import Animation
from utils.data import calc_bone_lengths
import generate
from utils.pre_run import EditOptions

TEST = False

def calc_attribute_score(attr, data_path, score_path, **kwargs):

    assert osp.isdir(data_path)
    assert osp.isdir(score_path)
    print('attr = {}'.format(attr))
    bvh_files = glob(osp.join(data_path, 'generated*.bvh'))
    W_all = None
    seeds = np.zeros((len(bvh_files)), dtype=int)
    score_all = np.zeros((len(bvh_files)))
    for file_idx, file in enumerate(bvh_files):
        anim, names, _ = BVH.load(file)
        joint_locations = Animation.positions_global(anim)
        joint_locations = joint_locations.transpose(1, 2, 0)  #  ==> joints, axes, frames
        bone_lengths = calc_bone_lengths(joint_locations, anim.parents, names)

        # calculate score
        score_all[file_idx] = eval(attr+'_score')(anim, names, joint_locations, bone_lengths, **kwargs)

        # save W at same order of score
        seeds[file_idx] = re.findall('\d+', osp.split(file)[1])[0]
        Wplus = np.load(osp.join(data_path, 'Wplus_'+str(seeds[file_idx])+'.npy'))
        assert (Wplus[0] == Wplus).all()
        if W_all is None:
            W_all = np.zeros(shape=(len(bvh_files), Wplus.shape[1]))
        W_all[file_idx] = Wplus[0]

        if (file_idx+1) % 1000 == 0:
            print('idx= {}, seed= {}, score= {}'.format(file_idx, seeds[file_idx], score_all[file_idx].round(2)))

    score_path = osp.join(score_path, attr+'_edit')
    os.makedirs(score_path, exist_ok=True)
    to_keep = ['scores', 'W', 'seeds']
    # file_names = {feature: '' for feature in to_keep}
    for feature, val in zip(to_keep, [score_all[:, np.newaxis], W_all, seeds]): # scores_all should be of size [n,1] (required by train_boundary)
        file = f'{attr}_{feature}.npy'
        np.save(osp.join(score_path, file), val)

    sorted_idx = score_all.argsort()
    mean = 0.5 * (score_all[sorted_idx[0]] + score_all[sorted_idx[-1]])
    argmean = np.abs(score_all-mean).argmin()
    extreme_idx = {'min': sorted_idx[0], 'max': sorted_idx[-1], 'med': int(sorted_idx.shape[0]/2), 'avg': argmean}
    with open(osp.join(score_path, 'extreme_idx.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=extreme_idx.keys(), delimiter='\t')
        writer.writeheader()
        writer.writerow(extreme_idx)
    for key, val in extreme_idx.items():
        print('{}: seed {}, value {}, index {}'.format(key, seeds[val], score_all[val].round(2), val))

    return score_path, score_all, W_all, seeds, extreme_idx



def verticality_score(anim, names, joint_locations, bone_lengths, **kwargs):
    # horizontal body vs. vertical
    ee_to_pelvis = {kwargs['chin']: 0, kwargs['r_ankle']: 0, kwargs['l_ankle']: 0}
    for ee in ee_to_pelvis.keys():
        name = ee
        while name != kwargs['pelvis']:
            ee_to_pelvis[ee] += bone_lengths['mean'][name]
            name = names[anim.parents[names.index(name)]]
    head_to_toe = ee_to_pelvis[kwargs['chin']] + (ee_to_pelvis[kwargs['r_ankle']] + ee_to_pelvis[kwargs['l_ankle']]) / 2
    chin_to_ankles_xyz = joint_locations[names.index(kwargs['chin'])] - joint_locations[
        [names.index(kwargs['r_ankle']), names.index(kwargs['r_ankle'])]]
    chin_to_ankles_z = chin_to_ankles_xyz[:,kwargs['vert_axis']]
    chin_to_furthest_ankle = np.abs(chin_to_ankles_z).max(axis=0) * np.sign(chin_to_ankles_z)
    verticality = chin_to_furthest_ankle.mean()  # signed
    verticality = verticality / head_to_toe  # normalize
    return verticality


def r_wrist_vert_score(anim, names, joint_locations, bone_lengths, **kwargs):

    # horizontal r_wrist vs. vertical
    r_elbow_to_wrist = joint_locations[names.index(kwargs['r_wrist'])] - joint_locations[names.index(kwargs['r_elbow'])]
    r_elbow_to_wrist_vert = r_elbow_to_wrist[kwargs['vert_axis']]
    verticality = r_elbow_to_wrist_vert.mean()  # signed
    verticality = verticality / bone_lengths['mean'][kwargs['r_wrist']]  # normalize
    return verticality


def calc_angle(anim, names, center_joint_name, end_joint_name, kwargs):
    """ compute the angle denoted by start_joint, center_joint, end_joint
        start_joint would be the parent of center_joint """

    # compute the angle in rest pose
    center_joint_idx = names.index(kwargs[center_joint_name])
    end_joint_idx = names.index(kwargs[end_joint_name])
    assert anim.parents[end_joint_idx] == center_joint_idx
    offset_to_start = -anim.offsets[center_joint_idx] # ray from center to start
    assert names[anim.parents[names.index(kwargs['r_wrist'])]] == kwargs['r_elbow']
    offset_to_end = anim.offsets[end_joint_idx] # ray from center to its end
    offset_to_start_normed = offset_to_start / (np.linalg.norm(offset_to_start) + 1e-20)
    offset_to_end_normed  = offset_to_end  / (np.linalg.norm(offset_to_end)  + 1e-20)
    rest_pose_angle_rad = np.arccos(np.inner(offset_to_start_normed, offset_to_end_normed))
    assert (np.inner(offset_to_end_normed, offset_to_end_normed)==np.sum(offset_to_end_normed*offset_to_end_normed)).all()
    rest_pose_angle_deg = np.rad2deg(rest_pose_angle_rad)
    assert ((rest_pose_angle_deg>=0) & (rest_pose_angle_deg<=360).all())

    # process the angle in the animation structure
    quat = anim.rotations[:, names.index(kwargs[center_joint_name])]
    # q==-q so abs does not change the rotation value but affects the axis_angle direction.
    # we use abs so rotation direction will be consistent.
    quat_avg = abs(quat.average())
    # quat_avg = quat.average()
    angle_rad, _ = quat_avg.angle_axis()
    angle_deg = np.rad2deg(angle_rad)
    assert ((angle_deg >=0) & (angle_deg <= 360)).all()
    angle_plus_rest_pose = angle_deg + rest_pose_angle_deg

    return angle_plus_rest_pose

def calc_angle_seq(anim, names, center_joint_name, end_joint_name, kwargs):
    """ compute the angles sequence denoted by start_joint, center_joint, end_joint
        start_joint would be the parent of center_joint """

    # compute the angle in rest pose
    center_joint_idx = names.index(kwargs[center_joint_name])
    end_joint_idx = names.index(kwargs[end_joint_name])
    assert anim.parents[end_joint_idx] == center_joint_idx
    offset_to_start = -anim.offsets[center_joint_idx] # ray from center to start
    assert names[anim.parents[names.index(kwargs[end_joint_name])]] == kwargs[center_joint_name]
    offset_to_end = anim.offsets[end_joint_idx] # ray from center to its end
    offset_to_start_normed = offset_to_start / (np.linalg.norm(offset_to_start) + 1e-20)
    offset_to_end_normed  = offset_to_end  / (np.linalg.norm(offset_to_end)  + 1e-20)
    rest_pose_angle_rad = np.arccos(np.inner(offset_to_start_normed, offset_to_end_normed))
    assert (np.inner(offset_to_end_normed, offset_to_end_normed)==np.sum(offset_to_end_normed*offset_to_end_normed)).all()
    rest_pose_angle_deg = np.rad2deg(rest_pose_angle_rad)
    assert ((rest_pose_angle_deg>=0) & (rest_pose_angle_deg<=360).all())

    # process the angle in the animation structure
    quat = anim.rotations[:, names.index(kwargs[center_joint_name])]
    # q==-q so abs does not change the rotation value but affects the axis_angle direction.
    # we use abs so rotation direction will be consistent.
    angle_degs = np.zeros(anim.shape[0])
    for idx ,q in enumerate(quat):
        quat_abs = abs(quat[idx])
        angle_rad, _ = quat_abs.angle_axis()
        angle_deg =np.rad2deg(angle_rad)
        assert ((angle_deg >= 0) & (angle_deg <= 360)).all()
        angle_degs[idx] = angle_deg

    angle_plus_rest_pose = angle_degs + rest_pose_angle_deg

    return angle_plus_rest_pose


# compute score for how monotonously raising the angle is
def get_monotonous_up_score(angles):
    i = 0
    avg = []
    window = 8
    step = 4
    s = 0
    while i + window <= len(angles):
        avg.append(np.mean(angles[i:i + window]))
        i += step
    for idx in range(1, len(avg)):
        if avg[idx]>avg[idx-1]:
            s+=1.0
    return s/len(avg)


def r_hand_lift_up_score(anim, names, joint_locations, bone_lengths, **kwargs):
    angle_plus_rest_pose_arm = calc_angle_seq(anim, names, 'r_arm', 'r_elbow', kwargs)
    angle_plus_rest_pose_shoulder = calc_angle_seq(anim, names, 'r_shoulder', 'r_arm', kwargs)

    # assert angle_plus_rest_pose_new == angle_plus_rest_pose

    angle_mod360 = np.mod(angle_plus_rest_pose_arm, 360)
    angle_reverse = 360 - angle_mod360  # the original rotation is measured towards the locked part of the elbow. reverse it to be more intuitive
    score1 = get_monotonous_up_score(angle_reverse)

    angle_mod360 = np.mod(angle_plus_rest_pose_shoulder, 360)
    angle_reverse = 360 - angle_mod360  # the original rotation is measured towards the locked part of the elbow. reverse it to be more intuitive
    score2 = get_monotonous_up_score(angle_reverse)

    score = np.mean([score1,score2])
    return score


def r_elbow_angle_score(anim, names, joint_locations, bone_lengths, **kwargs):
    angle_plus_rest_pose = calc_angle(anim, names, 'r_elbow', 'r_wrist', kwargs)

    angle_mod360 = np.mod(angle_plus_rest_pose, 360)
    angle_reverse = 360 - angle_mod360  # the original rotation is measured towards the locked part of the elbow. reverse it to be more intuitive
    return angle_reverse


def r_wrist_accel_score(anim, names, joint_locations, bone_lengths, **kwargs):
    r_wrist_loc_relative_to_r_elbow = joint_locations[names.index(kwargs['r_wrist'])] - joint_locations[names.index(kwargs['r_elbow'])]
    r_wrist_accel = r_wrist_loc_relative_to_r_elbow[:,:-2] - 2 * r_wrist_loc_relative_to_r_elbow[:,1:-1] + r_wrist_loc_relative_to_r_elbow[:,2:]
    r_wrist_accel = np.linalg.norm(r_wrist_accel, axis=0)
    r_wrist_accel = r_wrist_accel.mean()
    r_wrist_accel = r_wrist_accel / bone_lengths['mean'][kwargs['r_wrist']]  # normalize
    return r_wrist_accel


def latent_space_edit(args, model, entity, attributes, stages):
    if entity == 'Joint':
        kwargs = {'r_wrist': 'r_wrist', 'r_elbow': 'r_elbow', 'pelvis': 'pelvis', 'chin': 'chin',
                  'r_ankle': 'r_ankle', 'l_ankle': 'l_ankle', 'vert_axis': 2}
    else:
        kwargs = {'r_wrist': 'RightHand', 'r_elbow': 'RightForeArm', 'pelvis': 'Hips', 'chin': 'Head',
                  'r_arm': 'RightArm', 'r_shoulder': 'RightShoulder',
                  'r_ankle': 'RightFoot', 'l_ankle': 'LeftFoot', 'vert_axis': 1}

    # stage generate_motions
    data_path = generate_random_motions(args, model, stages)

    for attribute in attributes:
        print(f'***\n*** attribute = {attribute}\n***')

        # stage calc_score
        extreme_idx, latent_codes, score_path, scores, seeds = calc_score(attribute, data_path, kwargs, stages)

        # stage train_boundary
        boundary_path, boundary_normal, boundary_const = calc_boundary(latent_codes, score_path, scores, stages)

        # stage edit
        edit(args=args, boundary_path=boundary_path, extreme_idx=extreme_idx, model=model, seeds=seeds, score_path=score_path,
             stages=stages, entity=entity)

    pd.Series({'type': 'multi_edits', 'entity':entity}).to_csv(osp.join(data_path, 'args.csv'), sep='\t', header=None) # save args


def edit(args, boundary_path, extreme_idx, model, seeds, score_path, stages, entity):
    print('***\n*** Stage edit: Editing.\n***')
    if stages['edit'] is None:
        edit_types = ['min', 'max', 'avg']
        edit_idx = [extreme_idx[edit_type] for edit_type in edit_types]
        edit_idx = [int(val) for val in edit_idx]  # in case we read the indices from a file
        edit_seeds = seeds[edit_idx]
        edit_seeds = [str(seed) for seed in edit_seeds]
        edit_seeds = ','.join(edit_seeds)

        # compute edit radius
        boundary = np.load(osp.join(boundary_path, 'boundary.npy'), allow_pickle=True)[0]
        boundary_normal = boundary['normal']
        boundary_const = boundary['const']
        W_file = glob(osp.join(score_path, '*_W.npy'))
        assert len(W_file) == 1
        W_file = W_file[0]
        Ws = np.load(W_file)
        idx_min = int(extreme_idx['min'])
        idx_max = int(extreme_idx['max'])
        W_min = Ws[idx_min]
        W_max = Ws[idx_max]
        d_min = np.inner(W_min, boundary_normal) + boundary_const
        d_max = np.inner(W_max, boundary_normal) + boundary_const
        edit_radius = np.abs(np.array([d_min, d_max])).max() * 2
        if edit_radius > 2:
            edit_radius = edit_radius.round().astype(int)
        else:
            edit_radius = edit_radius.round(2)

        radii = [edit_radius, 10, 20, 40, 80, 160]
        for radius in radii:
            range_file_name = f'neg{radius}_{radius}'.replace('.', 'p')
            generate_params = ['--sample_seed', edit_seeds, '--boundary_path', osp.join(boundary_path, 'boundary.npy'),
                               '--out_path', osp.join(boundary_path, range_file_name), '--edit_radius',
                               str(radius),
                               '--ckpt', model, '--truncation', '1', '--type', 'edit', '--path', args.path]
            _ = generate.main(generate_params)

            range_file_name = f'neg{radius}_{radius}'.replace('.', 'p')
            extreme_types = ['min', 'max']
            if entity == 'Joint':
                kwargs = {'r_wrist': 'r_wrist', 'r_elbow': 'r_elbow', 'pelvis': 'pelvis', 'chin': 'chin',
                          'r_ankle': 'r_ankle', 'l_ankle': 'l_ankle', 'vert_axis': 2}
            else:
                kwargs = {'r_wrist': 'RightHand', 'r_elbow': 'RightForeArm', 'pelvis': 'Hips', 'chin': 'Head',
                          'r_ankle': 'RightFoot', 'l_ankle': 'LeftFoot', 'vert_axis': 1}
            for extreme_type in extreme_types:
                selected_idx = int(extreme_idx[extreme_type])
                min_files = glob(
                    osp.join(osp.join(boundary_path, range_file_name), f'generated_{seeds[selected_idx]}*.bvh'))
                print()
                print(f'scores for radius {radius}:')
                idx = [int(re.search('(\d\d).bvh', osp.split(f)[1]).group(1)) for f in min_files]
                sorted_idx = np.argsort(np.array(idx))
                for i, sorted_i in enumerate(sorted_idx):
                    file = min_files[sorted_i]
                    print(f'file = {osp.split(file)[1]}')
                    anim, names, _ = BVH.load(file)
                    score = r_elbow_angle_score(anim, names, None, None, **kwargs)
                    print(f'score = {score}')
                    W = Ws[selected_idx] + i * radius / 7 * boundary_normal
                    dist_from_boundary = np.inner(W, boundary_normal) + boundary_const
                    print(f'dist from boundary: {dist_from_boundary}')

    else:
        raise ('not supported yet')


def calc_boundary(latent_codes, score_path, scores, stages):
    print_str = '***\n*** Stage train_boundary: '
    ratio = 0.2
    boundary_path = osp.join(score_path, 'boundary_{}'.format(str(ratio).replace('.', 'p')))
    if stages['train_boundary'] is None:
        print(print_str + ' Training.\n***')
        boundary_normal, boundary_const = train_boundary(latent_codes=latent_codes, scores=scores, chosen_num_or_ratio=ratio)
        os.makedirs(boundary_path, exist_ok=True)
        np.save(osp.join(boundary_path, 'boundary.npy'), [{'normal': boundary_normal, 'const': boundary_const}])
    else:
        assert osp.isdir(boundary_path), 'Boundary folder does not exist.'
        boundary_normal, boundary_const = np.load(osp.join(boundary_path, 'boundary.npy'), allow_pickle=True)[0].values()
        print(print_str + f' Skipping. Using boundary from {boundary_path}\n***')
    return boundary_path, boundary_normal, boundary_const


def calc_score(attribute, data_path, kwargs, stages):
    print_str = '***\n*** Stage calc_score: '
    if stages['calc_score'] is None:
        print(print_str + 'Calculating score.\n***')
        score_path, scores, latent_codes, seeds, extreme_idx = calc_attribute_score(attribute, data_path, data_path,
                                                                                    **kwargs)
        scores = scores[:, np.newaxis]
    else:
        print(f'{print_str} Skipping. Using motions from {data_path}.\n***')
        score_path = osp.join(data_path, attribute + '_edit')
        file_suffices = ['scores', 'W', 'seeds']
        values = [None] * len(file_suffices)
        for i, suffix in enumerate(file_suffices):
            values[i] = np.load(osp.join(score_path, f'{attribute}_{suffix}.npy'), allow_pickle=True)
        scores, latent_codes, seeds = values
        with open(osp.join(score_path, 'extreme_idx.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            extreme_idx = next(reader)
    return extreme_idx, latent_codes, score_path, scores, seeds


def generate_random_motions(args, model, stages):
    # generate 10K images to calculate svm over
    print_str = '***\n*** Stage generate_motions: '
    if stages['generate_motions'] is None:
        n_motions = 10000
        if TEST: n_motions = 10
        print(print_str + f'Generating {n_motions} random motions.\n***')
        generate_params = ['--motions', str(n_motions), '--ckpt', model, '--truncation', '1', '--type', 'sample', '--path', args.path]
        data_path = generate.main(generate_params)
        print(f'   Generated motions saved to {data_path}')
    else:
        data_path = stages['generate_motions']
        print(print_str + f'Skipping. Using motions from {data_path}.\n***')
    return data_path


def train_boundary(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.02,
                   split_ratio=0.7,
                   invalid_value=None):
  """

  ************************************************************************************************************************
  BASED ON https://github.com/genforce/interfacegan/blob/8da3fc0fe2a1d4c88dc5f9bee65e8077093ad2bb/utils/manipulator.py#L12
  ************************************************************************************************************************

  Trains boundary in latent space with offline predicted attribute scores.

  Given a collection of latent codes and the attribute scores predicted from the
  corresponding images, this function will train a linear SVM by treating it as
  a bi-classification problem. Basically, the samples with highest attribute
  scores are treated as positive samples, while those with lowest scores as
  negative. For now, the latent code can ONLY be with 1 dimension.

  NOTE: The returned boundary is with shape (1, latent_space_dim), and also
  normalized with unit norm.

  Args:
    latent_codes: Input latent codes as training data.
    scores: Input attribute scores used to generate training labels.
    chosen_num_or_ratio: How many samples will be chosen as positive (negative)
      samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
      latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
      0.5 * latent_codes_num)` will be used. (default: 0.02)
    split_ratio: Ratio to split training and validation sets. (default: 0.7)
    invalid_value: This field is used to filter out data. (default: None)

  Returns:
    A decision boundary with type `numpy.ndarray`.

  Raises:
    ValueError: If the input `latent_codes` or `scores` are with invalid format.
  """
  if (not isinstance(latent_codes, np.ndarray) or
      not len(latent_codes.shape) == 2):
    raise ValueError(f'Input `latent_codes` should be with type'
                     f'`numpy.ndarray`, and shape [num_samples, '
                     f'latent_space_dim]!')
  num_samples = latent_codes.shape[0]
  latent_space_dim = latent_codes.shape[1]
  if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
      not scores.shape[0] == num_samples or not scores.shape[1] == 1):
    raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                     f'shape [num_samples, 1], where `num_samples` should be '
                     f'exactly same as that of input `latent_codes`!')
  if chosen_num_or_ratio <= 0:
    raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                     f'but {chosen_num_or_ratio} received!')

  print(f'Filtering training data.')
  if invalid_value is not None:
    latent_codes = latent_codes[scores[:, 0] != invalid_value]
    scores = scores[scores[:, 0] != invalid_value]

  print(f'Sorting scores to get positive and negative samples.')
  sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
  latent_codes = latent_codes[sorted_idx]
  scores = scores[sorted_idx]
  num_samples = latent_codes.shape[0]
  if 0 < chosen_num_or_ratio <= 1:
    chosen_num = int(num_samples * chosen_num_or_ratio)
  else:
    chosen_num = int(chosen_num_or_ratio)
  chosen_num = min(chosen_num, num_samples // 2)

  print(f'Spliting training and validation sets:')
  train_num = int(chosen_num * split_ratio)
  val_num = chosen_num - train_num
  # Positive samples.
  positive_idx = np.arange(chosen_num)
  np.random.shuffle(positive_idx)
  positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
  positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
  # Negative samples.
  negative_idx = np.arange(chosen_num)
  np.random.shuffle(negative_idx)
  negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
  negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
  # Training set.
  train_data = np.concatenate([positive_train, negative_train], axis=0)
  train_label = np.concatenate([np.ones(train_num, dtype=np.int),
                                np.zeros(train_num, dtype=np.int)], axis=0)
  print(f'  Training: {train_num} positive, {train_num} negative.')
  # Validation set.
  val_data = np.concatenate([positive_val, negative_val], axis=0)
  val_label = np.concatenate([np.ones(val_num, dtype=np.int),
                              np.zeros(val_num, dtype=np.int)], axis=0)
  print(f'  Validation: {val_num} positive, {val_num} negative.')
  # Remaining set.
  remaining_num = num_samples - chosen_num * 2
  remaining_data = latent_codes[chosen_num:-chosen_num]
  remaining_scores = scores[chosen_num:-chosen_num]
  decision_value = (scores[0] + scores[-1]) / 2
  remaining_label = np.ones(remaining_num, dtype=np.int)
  remaining_label[remaining_scores.ravel() < decision_value] = 0
  remaining_positive_num = np.sum(remaining_label == 1)
  remaining_negative_num = np.sum(remaining_label == 0)
  print(f'  Remaining: {remaining_positive_num} positive, '
              f'{remaining_negative_num} negative.')

  print(f'Training boundary.')
  clf = svm.SVC(kernel='linear')
  classifier = clf.fit(train_data, train_label)
  print(f'Finish training.')

  if val_num:
    val_prediction = classifier.predict(val_data)
    correct_num = np.sum(val_label == val_prediction)
    print(f'Accuracy for validation set: '
                f'{correct_num} / {val_num * 2} = '
                f'{correct_num / (val_num * 2):.6f}')

  if remaining_num:
    remaining_prediction = classifier.predict(remaining_data)
    correct_num = np.sum(remaining_label == remaining_prediction)
    print(f'Accuracy for remaining set: '
                f'{correct_num} / {remaining_num} = '
                f'{correct_num / remaining_num:.6f}')

  a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
  return a / np.linalg.norm(a), classifier.intercept_[0] / np.linalg.norm(a)


def main(args_not_parsed):
    parser = EditOptions()
    args = parser.parse_args(args_not_parsed)

    stages = {'generate_motions': args.data_path, 'calc_score': args.score_path, 'train_boundary': None, 'edit': None}
    latent_space_edit(args, args.model_path, args.entity, args.attr, stages)

if __name__ == "__main__":
    main(sys.argv[1:])
