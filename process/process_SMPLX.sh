
Model_path=${1:-"/ceph/hdd/yangsc21/Python/human_body_prior/support_data/dowloads/models/"}
Source_fold_1=${2:-'/ceph/datasets/BEAT/my_downsample'}
Source_fold_2=${3:-"/ceph/datasets/SMPLX/HumanML3D/"}

echo "Extract feature from SMPLX..."
echo "Path of SMPLX model:" $Model_path
echo "Path of BEAT dataset:" $Source_fold_1
echo "Path of HumanML3D dataset:" $Source_fold_2

python process_SMPLX.py \
    --model_path $Model_path \
    --source_fold $Source_fold_1/train/motion \
    --target_fold $Source_fold_1/train/motion_joints

python process_SMPLX.py \
    --model_path $Model_path \
    --source_fold $Source_fold_1/val/motion \
    --target_fold $Source_fold_1/val/motion_joints

python process_SMPLX.py \
    --model_path $Model_path \
    --source_fold $Source_fold_2/processed_motion \
    --target_fold $Source_fold_2/motion_joints_2

python motion_representation.py \
    --source_fold $Source_fold_1/val/motion_joints \
    --target_fold $Source_fold_1/val/motion_joints_vecs_3

python motion_representation.py \
    --source_fold $Source_fold_1/train/motion_joints \
    --target_fold $Source_fold_1/train/motion_joints_vecs_3

python motion_representation.py \
    --source_fold $Source_fold_2/motion_joints_2 \
    --target_fold $Source_fold_2/motion_joints_vecs_3
