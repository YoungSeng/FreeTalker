import os
import argparse
import pdb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx_folder', type=str, default=r"/ceph/datasets/SMPLX/")
    parser.add_argument('--processed_motion_path', type=str, default=r"/ceph/datasets/SMPLX/HumanML3D/motion_data/processed")
    parser.add_argument('--processed_text_path', type=str, default=r"/ceph/datasets/SMPLX/HumanML3D/text_data/processed")
    args = parser.parse_args()
    return args

def process_index(file, motion_folder, output_folder_motion, text_folder, output_folder_text):
    """
    source_path,start_frame,end_frame,new_name
    ./pose_data/KIT/3/kick_high_left02_poses.npy,0,117,000000.npy
    ...
    """

    with open(file, 'r') as f:
        lines = f.readlines()

    if not os.path.exists(output_folder_motion):
        os.makedirs(output_folder_motion)

    if not os.path.exists(output_folder_text):
        os.makedirs(output_folder_text)

    for line in lines[1:]:
        line = line.strip()
        source_path, start_frame, end_frame, new_name = line.split(',')
        # Remap source_path
        if 'humanact12' in source_path:
            continue
        elif 'MPI_HDM05' in source_path:
            source_path = source_path.replace('MPI_HDM05', 'HDM05')
            # continue
        elif 'Transitions_mocap' in source_path:
            source_path = source_path.replace('Transitions_mocap', 'Transitions')
        elif 'MPI_mosh' in source_path:
            source_path = source_path.replace('MPI_mosh', 'MoSh')
        elif 'BioMotionLab_NTroje' in source_path:
            source_path = source_path.replace('BioMotionLab_NTroje', 'BMLrub')
        elif 'DFaust_67' in source_path:
            source_path = source_path.replace('DFaust_67', 'DFaust')
        elif 'MPI_Limits' in source_path:
            source_path = source_path.replace('MPI_Limits', 'PosePrior')
        elif 'SSM_synced' in source_path:
            source_path = source_path.replace('SSM_synced', 'SSM')
        if 'MoSh' in source_path or 'ACCAD' in source_path:
            source_path = source_path.replace(' ', '_')
        print(f'Processing：{new_name}')
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        new_name = new_name.replace('.npy', '')

        if os.path.exists(os.path.join(output_folder_motion, new_name + '.npz')):
            print(f'File {new_name} already exists.')
            continue
        else:
            source_path = os.path.join(motion_folder, source_path.replace('./pose_data/', '').replace('_poses.npy', '_stageii.npz'))
            source_File_exist = True
            if not os.path.exists(source_path):
                if 'Eyes_Japan_Dataset' in source_path:
                    if os.path.exists(source_path.replace(' ', '_')):
                        source_path = source_path.replace(' ', '_')
                    elif os.path.exists(source_path.replace(' ', '-')):
                        source_path = source_path.replace(' ', '-')
                    else:
                        source_File_exist = False
                else:
                    source_File_exist = False

                if source_File_exist == False:
                    print(f'File {source_path} does not exist.')
                    continue
                else:
                    text_path = os.path.join(text_folder, new_name + '.txt')
                    if not os.path.exists(text_path):
                        print(f'File {text_path} does not exist.')
                        continue
                    rename_path = os.path.join(output_folder_motion, new_name + '.npz')
                    os.system(f'cp "{source_path}" "{rename_path}"')
                    os.system(f'cp "{text_path}" "{output_folder_text}"')
                    print(f'File {rename_path} copied.')

            text_path = os.path.join(text_folder, new_name + '.txt')
            if not os.path.exists(text_path):
                print(f'File {text_path} does not exist.')
                continue
            rename_path = os.path.join(output_folder_motion, new_name + '.npz')
            os.system(f'cp "{source_path}" "{rename_path}"')
            os.system(f'cp "{text_path}" "{output_folder_text}"')
            print(f'File {rename_path} copied.')


if __name__ == '__main__':

    args = get_args()
    smplx_folder = args.smplx_folder
    processed_motion_path = args.processed_motion_path
    processed_text_path = args.processed_text_path

    index_file = 'index.csv'
    text_folder = 'texts'
    process_index(index_file, smplx_folder, processed_motion_path, text_folder, processed_text_path)

    print('Done.')
