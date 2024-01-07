
import os
import json
import subprocess

root = '/apdcephfs/private_yyyyyyyang/code/mdm/run_jizhi'
mirrors = 'mirrors.tencent.com/mingzhenzhu/torch1.7.1-cu102:mst2'
name = 'my_v3_4'

root = os.path.join(root, name)
os.makedirs(root, exist_ok=True)

# "business_flag": "TEG_AILab_CVC_chongqing",

config = {
    "Token": "Jtcs_U_nX9crYq8WKvEfIw",
    "business_flag": "TEG_AILab_CVC_DigitalContent",
    "model_local_file_path": root,
    "priority_level": "HIGH",
    "host_num": 1,
    "host_gpu_num": 1,
    "image_full_name": mirrors,
    "mount_ceph_business_flag": "TEG_AILab_CVC_chongqing",
    "task_flag": 'my_v3_4'
}

elastic_config = {
    "Token": "Jtcs_U_nX9crYq8WKvEfIw",
    "business_flag": "TEG_AILab_CVC_DigitalContent",
    "model_local_file_path": root,
    "priority_level": "LOW",
    "elastic_level": 1,
    "host_num": 1,
    "host_gpu_num": 1,
    "image_full_name": mirrors,
    "mount_ceph_business_flag": "TEG_AlLab_CVC_chongqing",
    "task_flag": 'v0_3'
}


with open(os.path.join(root, 'config.json'), 'w') as f:
    json.dump(config, f)

with open(os.path.join(root, 'elastic_config.json'), 'w') as f:
    json.dump(elastic_config, f)

cmd_lines = [
    'echo "start.."\n',
    'export PATH=/apdcephfs/private_yyyyyyyang/miniconda3/envs/mdm/bin:$PATH\n',
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apdcephfs/private_yyyyyyyang/miniconda3/lib/\n'
    'cd /apdcephfs/private_yyyyyyyang/code/mdm\n',
    # 'pip list\n',
    # 'lspci | grep -i nvidia\n',
    f'/apdcephfs/private_yyyyyyyang/miniconda3/envs/mdm/bin/python -m train.train_mdm --save_dir save/my_v3_4 --overwrite --batch_size 256 --n_frames 180 --n_seed 0\n',
    'echo "train done.."\n'
    ]

# cmd_lines = [
#     'echo "start.."\n',
#     'export PATH=/apdcephfs/private_yyyyyyyang/miniconda3/envs/human_body_prior/bin:$PATH\n',
#     # 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apdcephfs/private_yyyyyyyang/miniconda3/lib/\n'
#     'cd /apdcephfs/private_yyyyyyyang/code/human_body_prior/tutorials\n',
#     # 'pip list\n',
#     # 'lspci | grep -i nvidia\n',
#     f'/apdcephfs/private_yyyyyyyang/miniconda3/envs/human_body_prior/bin/python "/apdcephfs/private_yyyyyyyang/code/mdm/process/mdm_to_smplx.py"\n',
#     'echo "train done.."\n'
#     ]

with open(os.path.join(root, 'start.sh'), 'w') as f:
    f.writelines(cmd_lines)

# taiji_client start --scfg elastic_config.json

os.chdir(root)
print(os.getcwd())
# subprocess.run(['taiji_client', 'start', '--scfg', 'elastic_config.json'])
subprocess.run(['taiji_client', 'start', '--scfg', 'config.json'])
