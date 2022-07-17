import numpy as np
import pickle as pkl
import os
import argparse
humanact12_coarse_action_to_label = {x: x-1 for x in range(1, 13)}

def splitname(name):
    subject = name[1:3]
    group = name[4:6]
    time = name[7:9]
    frame1 = name[10:14]
    frame2 = name[15:19]
    action = name[20:24]
    return subject, group, time, frame1, frame2, action

humanact12_coarse_action_enumerator = {
    1: "warm_up",
    2: "walk",
    3: "run",
    4: "jump",
    5: "drink",
    6: "lift_dumbbell",
    7: "sit",
    8: "eat",
    9: "turn steering wheel",
    10: "phone",
    11: "boxing",
    12: "throw",
}


def get_action(name, coarse=True):
    subject, group, time, frame1, frame2, action = splitname(name)
    if coarse:
        return action[:2]
    else:
        return action

def process_data(savepath, datapath, savepath_uniform):
    data_list = os.listdir(datapath)
    data_list.sort()
    lengths = []
    dataset = {"joints3D": [], "y": [], "joints3D_sampled": []}
    dataset_uniform = {"y": [], "joints3D_sampled": []}
    for index, name in enumerate(data_list):
        joints3D_orig = np.load(os.path.join(datapath, name))
        dataset['joints3D'].append(joints3D_orig)
        # match joints of skeleton to joints used by styleMotion (15 joints)
        idx = [15, 12, 16, 18, 20, 17, 19, 21, 0, 1, 4, 7, 2, 5, 8]
        joints3D_sampled = joints3D_orig[:, idx, :]
        dataset['joints3D_sampled'].append(joints3D_sampled)

        action = get_action(name, coarse=True)
        label = humanact12_coarse_action_to_label[int(action)]
        dataset['y'].append(label)

    dataset_fixed_length = []

    frames = 64
    for idx, motion in enumerate(dataset["joints3D_sampled"]):
        length = motion.shape[0]
        if length < frames:
            continue
        if length == frames:
            m = np.zeros(np.array([frames, 16, 3]))
            m[:,0:15,:] = motion
            dataset_fixed_length.append(m)
            dataset_uniform['joints3D_sampled'].append(motion)
            dataset_uniform['y'].append(dataset['y'][idx])
        i = 0
        while i+frames < length:
            m = np.zeros(np.array([frames, 16, 3]))
            m[:,0:15,:] = motion[i:i+64]
            dataset_fixed_length.append(m)
            dataset_uniform['y'].append(dataset['y'][idx])
            dataset_uniform['joints3D_sampled'].append(motion[i:i+64])
            i+=frames//2

    ds = np.asarray(dataset_fixed_length)
    ds = ds.transpose(0,2,3,1)

    np.save('data/humanact12motion.npy', ds)

    pkl.dump(dataset_uniform, open(savepath_uniform, "wb"))

    pkl.dump(dataset, open(savepath, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the generator and compute action recognition model features")
    parser.add_argument("--datapath", type=str, help="path to data folder")
    args = parser.parse_args()

    folder = "data/"
    os.makedirs(folder, exist_ok=True)
    savepath = os.path.join(folder, "humanact12.pkl")
    savepath_uniform = os.path.join(folder, "humanact12uniform.pkl")
    process_data(savepath, args.datapath, savepath_uniform)