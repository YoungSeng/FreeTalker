
Step=${1:-'prepare'}
BEAT_path=${2:-'/apdcephfs/share_1290939/new_data/BEAT'}
HumanML3D_path=${3:-"/ceph/datasets/SMPLX/HumanML3D"}
WavLM_path=${4:-"/ceph/hdd/yangsc21/Python/UnifiedGesture/diffusion_latent/wavlm_cache/WavLM-Large.pt"}
CLIP_path=${5:-"/ceph/hdd/yangsc21/Python/mdm/data/clip"}
output_path=${6:-"/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data"}

echo "Processing dataset..."
echo "Step" $Step
echo "Path of BEAT dataset:" $BEAT_path
echo "Path of HumanML3D dataset:" $HumanML3D_path
echo "Path of WavLM:" $WavLM_path
echo "Path of CLIP:" $CLIP_path
echo "Output path:" $output_path

python process_BEAT_dataset.py \
    --step $Step \
    --BEAT_smplx_path $BEAT_path/my_smplx \
    --BEAT_txt_save_PATH $BEAT_path \
    --wavlm_model_path $WavLM_path \
    --BEAT_source_PATH $BEAT_path/beat_english_v0.2.1/ \
    --BEAT_save_PATH $BEAT_path/my_wav_feat \
    --h5_save_path $output_path \
    --down_sample_save_path $BEAT_path/my_downsample/ \
    --HUMANML3D_smplx_path $HumanML3D_path/motion_joints_vecs_3 \
    --HUMANML3D_save_path $HumanML3D_path/v3_HUMANML3D_txt_feat \
    --HUMANML3D_txt_save_path $HumanML3D_path \
    --clip_model_path $CLIP_path