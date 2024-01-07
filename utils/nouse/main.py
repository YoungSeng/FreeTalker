#
# import os
# import json
# import subprocess
#
# root = '/apdcephfs_cq3/share_1290939/yyyyyyyang/code/mdm/run_jizhi'
# mirrors = 'mirrors.tencent.com/watsonswang/python3.7-cuda11.0:latest'
#
# name = 'mdm_kit'
#
# root = os.path.join(root, name)
# os.makedirs(root, exist_ok=True)
#
#
# config = {
#     "Token": "Jtcs_U_nX9crYq8WKvEfIw",
#     "business_flag": "TEG_AILab_CVC_chongqing",
#     "model_local_file_path": root,
#     "priority_level": "HIGH",
#     "host_num": 1,
#     "host_gpu_num": 1,
#     "image_full_name": mirrors
# }
#
# elastic_config = {
#     "Token": "Jtcs_U_nX9crYq8WKvEfIw",
#     "business_flag": "TEG_AILab_CVC_chongqing",
#     "model_local_file_path": root,
#     "priority_level": "LOW",
#     "elastic_level": 1,
#     "host_num": 1,
#     "host_gpu_num": 1,
#     "image_full_name": mirrors
# }
#
#
# with open(os.path.join(root, 'config.json'), 'w') as f:
#     json.dump(config, f)
#
# with open(os.path.join(root, 'elastic_config.json'), 'w') as f:
#     json.dump(elastic_config, f)
#
# cmd_lines = [
#     'echo "start.."\n',
#     'export PATH=/apdcephfs_cq3/share_1290939/yyyyyyyang/miniconda3/bin:$PATH\n',
#     'cd\n',
#     'pwd\n',
#     'sudo cd \n',
#     'pwd\n',
#     # 'sudo cd /apdcephfs_cq3/share_1290939/yyyyyyyang/code/mdm\n',
#     # f'/apdcephfs_cq3/share_1290939/yyyyyyyang/miniconda3/envs/mdm/bin/python -m train.train_mdm --save_dir save/my_kit_trans_enc_512_ --dataset kit --overwrite\n',
#     'sudo cd /root/code/motion-diffusion-model/\n',
#     f'/apdcephfs_cq3/share_1290939/yyyyyyyang/miniconda3/envs/mdm/bin/python -m train.train_mdm --save_dir save/my_kit_trans_enc_512_ --dataset kit --overwrite\n',
#     'echo "train done.."\n'
#     ]
#
# with open(os.path.join(root, 'start.sh'), 'w') as f:
#     f.writelines(cmd_lines)
#
# # taiji_client start --scfg elastic_config.json
#
# os.chdir(root)
# print(os.getcwd())
# # subprocess.run(['taiji_client', 'start', '--scfg', 'elastic_config.json'])
# subprocess.run(['taiji_client', 'start', '--scfg', 'config.json'])
#
# # /apdcephfs_cq3/share_1290939/yyyyyyyang 加上sudo也访问不了

import subprocess
video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/positions_real.mp4"
source_BEAT_path = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/"
output_video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/positions_real-with-audio.mp4"
audio_tag = [['2', '2_scott_0_55_55', 0, 125], ['2', '2_scott_0_55_55', 125, 125 + 125],
             ['2', '2_scott_0_55_55', 125 * 2, 125 * 3]]
merge_segment = [[190, 190 + 125], [190 + 125, 190 + 125 * 2], [190 + 125 * 2, 190 + 125 * 3]]
subprocess.run(
    f'python -m process.merge_mp4_audio --video_file {video_file} --source_BEAT_path {source_BEAT_path} --output_video_file {output_video_file} --audio_tag {audio_tag[0]} --merge_segment {merge_segment[0]}', shell=True)
# subprocess.run(
#     f'python -m process.merge_mp4_audio --video_file {output_video_file} --source_BEAT_path {source_BEAT_path} --output_video_file {output_video_file} --audio_tag {audio_tag[0]} --merge_segment {merge_segment[0]}', shell=True)
# subprocess.run(
#     f'python -m process.merge_mp4_audio --video_file {output_video_file} --source_BEAT_path {source_BEAT_path} --output_video_file {output_video_file} --audio_tag {audio_tag[0]} --merge_segment {merge_segment[0]}', shell=True)

