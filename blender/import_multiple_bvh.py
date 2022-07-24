import bpy
import glob
import os.path as osp
import numpy as np
import sys
# import pandas as pd
import re
import csv
import os

def location(horiz_loc, vert_loc, entity):
    if entity.lower() == 'joint':
        location = (horiz_loc, 0, vert_loc)
    else:
        location = (horiz_loc, vert_loc, 0)
    return location

scale_factor = 3

base_path = '<type the path to your models>'               # example: '/home/username/train_outputs'
cur_path = '<type the rest of the path>'  # example: 'experiment_name.362db4a171934333bea96e9c10712d95/models/079999_files/22_06_16_15_10_sample_10')'
path = osp.join(base_path, cur_path)
files = glob.glob(osp.join(path, 'generated_*.bvh'))
files = [osp.basename(file) for file in files if 'fixed' not in file]  # use only file name without path
files.sort()
n_files = len(files)

# header
try:
    with open(osp.join(path, 'args.csv')) as f:
        reader = csv.reader(f, delimiter='\t')
        args = {row[0]: row[1] for row in reader}
except:
    print()
    print('***********************')
    print('args.csv was not found.')
    print('***********************')
    args = {'type': 'sample'}

if 'entity' in args:
    entity = args['entity']
else:
    entity = 'Edge'

if entity.lower() == 'joint':
    col_mult = 1
    text_rot = (1.5708, 0, 0) # 90 degrees
else:
    col_mult = 80
    text_rot = (0, 0, 0)
row_mult = col_mult * 2

z_loc_for_folder_name = 0

def interp(origin_x=0, origin_z=0):
    global z_loc_for_folder_name
    z_loc_for_folder_name = origin_z + 2 * col_mult

    # add title
    global files, n_files
    headers = args['interp_seeds']
    headers = re.findall('\[\d+,* *\d*\]', headers)
    for i, header in enumerate(headers):
        text_loc = location(origin_x, (origin_z + .8 * col_mult) - i * scale_factor * col_mult, entity)
        header = re.findall('\d+', header)
        bpy.ops.object.text_add(enter_editmode=False, align='WORLD',
                                location=text_loc, scale=(1, 1, 1))
        bpy.context.object.rotation_euler = text_rot
        seed_from = header[0]
        seed_to = header[1] if len(header) > 1 else 'mean'
        bpy.context.object.data.body = seed_from + '-' + seed_to
        scale = 0.5 * col_mult
        bpy.context.object.scale = [scale, scale, scale]

    # add motions
    for k, header in enumerate(headers):
        header = re.findall('\d+', header)
        folder_name = 'interp_' + header[0] + '-' + (header[1] if len(header) > 1 else 'mean')
        folder = osp.join(path, folder_name)
        files = glob.glob(osp.join(folder, '*.bvh'))
        files = [osp.basename(file) for file in files]
        n_files = len(files)
        parses = [re.findall('\d+', file) for file in files]
        order = np.array(parses).astype(int).flatten()
        assert order.shape[0] == n_files  # in case there were digits in file name, use parses[[-1]]
        sorted_order_idx = np.argsort(order)
        for i, ord in enumerate(sorted_order_idx):
            x_loc = origin_x + i * col_mult
            z_loc = origin_z
            bpy.ops.import_anim.bvh(filepath=osp.join(folder, files[ord]), axis_forward='Y', axis_up='Z')
            bpy.context.object.location[0] = x_loc
            if entity.lower() == 'joint':
                bpy.context.object.location[2] = z_loc - k * scale_factor
            else:
                bpy.context.object.location[1] = z_loc - k * scale_factor * col_mult

def sample():
    global z_loc_for_folder_name
    # simple case
    n_cols = np.ceil(np.sqrt(n_files / 2)).astype(int)
    n_rows = np.floor(n_files / n_cols)
    # print(n_files, n_rows, n_cols)

    seeds = [re.findall('\d+', file) for file in files]
    seeds = np.array(seeds).astype(int).flatten()
    z_loc_for_folder_name = np.ceil(n_files / n_rows).astype(int) * row_mult

    for i in range(n_files):
        bpy.ops.import_anim.bvh(filepath=osp.join(path, files[i]), axis_forward='Y', axis_up='Z')
        x_loc = (i % n_rows) * col_mult
        z_loc = np.floor(i / n_rows).astype(int) * row_mult
        # print(i, x_loc, z_loc)

        text_vert_loc = z_loc + 0.35 * row_mult
        if entity.lower() == 'joint':
            bpy.context.object.location[0] = x_loc
            bpy.context.object.location[2] = z_loc
            text_loc = (x_loc, 0, text_vert_loc)
        else:
            bpy.context.object.location[0] = x_loc
            bpy.context.object.location[1] = z_loc
            text_loc = (x_loc, text_vert_loc, 0)

        bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=text_loc, scale=(1, 1, 1))
        bpy.context.object.rotation_euler = text_rot
        bpy.context.object.data.body = '_'.join(files[i].split('_')[1:])[:-4]
        text_scale = 0.2 * col_mult
        bpy.context.object.scale = [text_scale, text_scale, text_scale]
        if i % 10 == 0:
            print(f'#{i} of {n_files}')

def edit(files, n_files, cur_path, origin_x=0, sub_folder=None):
    global z_loc_for_folder_name
    order = np.argsort(np.array(files))
    files = np.array(files)[order]

    indices = [re.findall('\d+', file) for file in files]
    indices = np.array(indices).astype(int)
    unordered_seeds = np.unique(indices[:, 0])
    interps_per_seed = int(n_files / len(unordered_seeds))
    seeds = indices[::interps_per_seed, 0]

    file_idx = 0
    for seed_idx, seed in enumerate(seeds):
        while file_idx < indices.shape[0] and seed == indices[file_idx, 0]:
            offset_idx = indices[file_idx, 1]
            bpy.ops.import_anim.bvh(filepath=osp.join(cur_path, files[file_idx]), axis_forward='Y', axis_up='Z')
            x_loc = origin_x + offset_idx * col_mult * 1.5
            z_loc = seed_idx * row_mult
            bpy.context.object.location = location(x_loc, z_loc, entity)

            bpy.ops.object.text_add(enter_editmode=False, align='WORLD',
                                    location=location(x_loc - 0.5 * col_mult, z_loc + 0.7 * col_mult, entity),
                                    scale=(1, 1, 1))
            bpy.context.object.rotation_euler = text_rot
            bpy.context.object.data.body = '{}_{}'.format(seed, offset_idx)
            bpy.context.object.scale = [0.2*col_mult, 0.2*col_mult, 0.2*col_mult]

            file_idx += 1

    max_z = len(seeds)*row_mult - 0.5 * col_mult

    if sub_folder is not None:
        # being called from multi_edits
        # text sub folder name
        bpy.ops.object.text_add(enter_editmode=False, align='WORLD',
                                location=location(origin_x, max_z, entity), scale=(1, 1, 1))
        bpy.context.object.rotation_euler = text_rot
        bpy.context.object.data.body = sub_folder
        bpy.context.object.scale = [0.2 * col_mult, 0.2 * col_mult, 0.2 * col_mult]

    return x_loc, max_z

def multi_edits():
    global z_loc_for_folder_name
    z_loc_for_folder_name = 0
    origin_x = 0
    for root, dirs, files in os.walk(path, topdown=False):
        print('multi_edits root, dirs:', root, dirs)

        if 'args.csv' in files and root != path:
            sub_folder = osp.relpath(root, path)

            # sanity check
            with open(osp.join(root, 'args.csv')) as f:
                reader = csv.reader(f, delimiter='\t')
                inner_args = {row[0]: row[1] for row in reader}
                assert inner_args['type'] == 'edit'

            inner_files = glob.glob(osp.join(root, '*.bvh'))
            inner_files = [osp.basename(file) for file in inner_files]  # use only file name without path
            max_x, max_z = edit(inner_files, len(inner_files), root, origin_x, sub_folder)

            z_loc_for_folder_name = max(max_z, z_loc_for_folder_name)
            origin_x = max_x + 3*col_mult

    z_loc_for_folder_name += 0.7 * col_mult
    bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location= location(1, z_loc_for_folder_name + col_mult, entity), scale=(1, 1, 1))
    bpy.context.object.rotation_euler = text_rot
    bpy.context.object.data.body = 'Edits'
    text_scale = col_mult
    bpy.context.object.scale = [text_scale, text_scale, text_scale]


print('type = {}'.format(args['type']))

if args['type'] == 'interp':
    interp()

elif args['type'] == 'sample':
    # simple case
    sample()

elif args['type'] == 'edit':
    edit(files, n_files, path)

elif args['type'] == 'multi_edits':
    multi_edits()

if 'n_frames_dataset' in args and args['n_frames_dataset'].isdigit():
    bpy.context.scene.frame_end = int(args['n_frames_dataset'])
else:
    bpy.context.scene.frame_end = 64

# text folder name
bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=location(0, 
                        z_loc_for_folder_name, entity), scale=(1, 1, 1))
bpy.context.object.rotation_euler = text_rot
bpy.context.object.data.body = osp.relpath(path, base_path)  # shorten trivial part of path to reduce clutter

# if path contains a long hexa number (clearml), shorten it to three initial and ending characters only
bpy.context.object.data.body = re.sub('(.*\.[a-f\d]{3})[a-f\d]+([a-f\d]{3})', '\g<1>_\g<2>', bpy.context.object.data.body)
bpy.context.object.data.body = re.sub('models', '\nmodels', bpy.context.object.data.body) # break line
bpy.context.object.scale = [0.4*col_mult, 0.4*col_mult, 0.4*col_mult]

