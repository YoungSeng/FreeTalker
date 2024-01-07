from bvh2smplx import bvh_to_smplx, save_npz
import os
import argparse
import pdb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_BEAT_path', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1")
    parser.add_argument('--save_BEAT_smplx_path', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/my_smplx")
    args = parser.parse_args()
    return args

def main(source_BEAT_path, save_BEAT_smplx_path):

    if not os.path.exists(save_BEAT_smplx_path):
        os.makedirs(save_BEAT_smplx_path)

    for speaker in ['2', '4', '6', '8']:        # os.listdir(source_BEAT_path)
        if not os.path.exists(os.path.join(save_BEAT_smplx_path, speaker)):
            os.makedirs(os.path.join(save_BEAT_smplx_path, speaker))

        for bvh_file in os.listdir(os.path.join(source_BEAT_path, speaker)):
            if not bvh_file.endswith('.bvh'):
                continue
            print(bvh_file)
            smplx_trans, smplx_poses = bvh_to_smplx(os.path.join(source_BEAT_path, speaker, bvh_file))
            output_file = os.path.join(save_BEAT_smplx_path, speaker, bvh_file[:-4] + '.npz')
            save_npz(output_file, smplx_trans, smplx_poses, frame_rate=20)


if __name__ == '__main__':
    '''
    python BEAT2smplx.py --source_BEAT_path /ceph/datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/ --save_BEAT_smplx_path /ceph/datasets/BEAT/my_smplx
    '''

    args = get_args()
    source_BEAT_path = args.source_BEAT_path
    save_BEAT_smplx_path = args.save_BEAT_smplx_path

    main(source_BEAT_path, save_BEAT_smplx_path)
