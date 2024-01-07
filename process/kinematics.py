import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# SMPLX关节名称
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip", #
    "spine1",
    "left_knee",
    "right_knee", # 5
    "spine2",
    "left_ankle", # 7
    "right_ankle", # 8
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",   # 15
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",    # 22
    "left_eye_smplhf",
    "right_eye_smplhf", # 24
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3", # 30
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3", # 36
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",   # 42
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",   # 48
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3"]


# 定义左脚关节链
left_leg_chain = Chain(name='left_leg', links=[
    OriginLink(),
    URDFLink(
        name=JOINT_NAMES[0],
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    ),
    URDFLink(
        name=JOINT_NAMES[1],
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    ),
    URDFLink(
        name=JOINT_NAMES[4],
        origin_translation=[0, 0, -1],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    ),
    URDFLink(
        name=JOINT_NAMES[7],
        origin_translation=[0, 0, -1],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    )
])

# 定义右脚关节链
right_leg_chain = Chain(name='right_leg', links=[
    OriginLink(),
    URDFLink(
        name=JOINT_NAMES[0],
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    ),
    URDFLink(
        name=JOINT_NAMES[2],
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    ),
    URDFLink(
        name=JOINT_NAMES[5],
        origin_translation=[0, 0, -1],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    ),
    URDFLink(
        name=JOINT_NAMES[8],
        origin_translation=[0, 0, -1],
        origin_orientation=[0, 0, 0],
        axis=[0, 0, 0],
    )
])

# 其他代码保持不变

# 目标左脚位置（3D position）
target_left_position = [0, 0, -2]

# 计算左脚关节角度
left_joint_angles = left_leg_chain.inverse_kinematics(target_left_position)

# 输出左脚关节角度
print("左脚关节角度：", left_joint_angles)

# 目标右脚位置（3D position）
target_right_position = [0, 0, -2]

# 计算右脚关节角度
right_joint_angles = right_leg_chain.inverse_kinematics(target_right_position)

# 输出右脚关节角度
print("右脚关节角度：", right_joint_angles)


'''
python process/kinematics.py
'''
