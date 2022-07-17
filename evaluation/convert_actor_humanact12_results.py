import numpy as np
import argparse
import os


def process_data(in_path, out_path):
    res = {"joints3D": [], "joints3D_sampled": []}
    joints3D_orig = np.load(os.path.join(in_path))
    res['joints3D'].append(joints3D_orig)
    # match joints of skeleton to joints used by styleMotion (15 joints)
    idx = [15, 12, 16, 18, 20, 17, 19, 21, 0, 1, 4, 7, 2, 5, 8]
    joints3D_sampled = joints3D_orig[:, idx, :]
    res['joints3D_sampled'].append(joints3D_sampled)

    ds = np.asarray(joints3D_sampled)
    np.save(out_path, ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the generator and compute action recognition model features")
    parser.add_argument("--in_path", type=str, help="path to input file")
    parser.add_argument("--out_path", type=str, help="path to output file")
    args = parser.parse_args()

    process_data(args.in_path, args.out_path)