import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx_folder', type=str, default=r"/ceph/datasets/SMPLX/")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    smplx_folder = args.smplx_folder
    for item in os.listdir(smplx_folder):
        if item.endswith('.tar.bz2'):
            print(item)
            os.system(f'tar -xjvf {os.path.join(smplx_folder, item)} -C {smplx_folder}')

    print('Done.')
