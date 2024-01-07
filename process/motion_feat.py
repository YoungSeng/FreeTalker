import pdb

import torch
import torchgeometry as tgm


def foot_contacts(joints):
    left_heel = joints[:, 62, :]
    right_heel = joints[:, 65, :]
    left_toe = joints[:, 60, :]
    right_toe = joints[:, 63, :]

    left_contact = torch.logical_or(left_heel[:, 1] < 0.05, left_toe[:, 1] < 0.05)
    right_contact = torch.logical_or(right_heel[:, 1] < 0.05, right_toe[:, 1] < 0.05)

    return left_contact, right_contact


def quaternion_representation(joints):
    pelvis = joints[:, 0, :]
    left_hip = joints[:, 1, :]
    right_hip = joints[:, 2, :]

    left_hip_rot = tgm.angle_axis_to_quaternion(left_hip - pelvis)
    right_hip_rot = tgm.angle_axis_to_quaternion(right_hip - pelvis)

    return left_hip_rot, right_hip_rot


def root_rotation_and_linear_velocity(joints):
    pelvis = joints[:, 0, :]
    pelvis_rot = tgm.angle_axis_to_quaternion(pelvis)
    pelvis_vel = torch.cat([torch.zeros(1, 3), pelvis[1:] - pelvis[:-1]], dim=0)

    return pelvis_rot, pelvis_vel


def joint_rotation_invariant_position_representation(joints):
    pelvis = joints[:, 0, :]
    left_hip = joints[:, 1, :]
    right_hip = joints[:, 2, :]

    left_hip_pos = left_hip - pelvis
    right_hip_pos = right_hip - pelvis

    return left_hip_pos, right_hip_pos


def joint_angles(joints):
    joint_angles = [torch.zeros(joints.shape[0])]
    for i in range(1, joints.shape[1] - 1):
        parent = joints[:, i - 1, :]
        joint = joints[:, i, :]
        child = joints[:, i + 1, :]
        vec1 = joint - parent
        vec2 = child - joint
        vec1 = vec1 / torch.norm(vec1, dim=1, keepdim=True)
        vec2 = vec2 / torch.norm(vec2, dim=1, keepdim=True)
        angle = torch.acos(torch.clamp(torch.sum(vec1 * vec2, dim=1), -1, 1))
        joint_angles.append(angle)
    joint_angles.append(torch.zeros(joints.shape[0]))  # Add a zero for the last joint angle
    joint_angles = torch.stack(joint_angles, dim=1)

    return joint_angles



def joint_rotation_representation(joints):
    joint_rotations = [torch.zeros(joints.shape[0], 4)]
    for i in range(1, joints.shape[1]):
        parent = joints[:, i - 1, :]
        child = joints[:, i, :]
        rotation = tgm.angle_axis_to_quaternion(child - parent)
        joint_rotations.append(rotation)
    joint_rotations = torch.stack(joint_rotations, dim=1)

    return joint_rotations


def joint_velocity_representation(joints):
    joint_velocities = torch.cat([torch.zeros(1, joints.shape[1], 3), joints[1:] - joints[:-1]], dim=0)

    return joint_velocities


def extract_motion_features(joints):
    left_contact, right_contact = foot_contacts(joints)
    left_hip_rot, right_hip_rot = quaternion_representation(joints)
    pelvis_rot, pelvis_vel = root_rotation_and_linear_velocity(joints)
    left_hip_pos, right_hip_pos = joint_rotation_invariant_position_representation(joints)
    joint_angles_arr = joint_angles(joints)
    joint_rotations = joint_rotation_representation(joints)
    joint_velocities = joint_velocity_representation(joints)

    # Add extra dimension for 1D tensors
    left_contact = left_contact[:, None].to(torch.float32)
    right_contact = right_contact[:, None].to(torch.float32)

    # Convert 3D tensors to 2D tensors
    joint_angles_arr = joint_angles_arr.view(joint_angles_arr.shape[0], -1)
    joint_rotations = joint_rotations.view(joint_rotations.shape[0], -1)
    joint_velocities = joint_velocities.view(joint_velocities.shape[0], -1)

    # print('left_contact', left_contact,
    #       'right_contact', right_contact,
    #       'left_hip_rot', left_hip_rot,
    #       'right_hip_rot', right_hip_rot,
    #       'pelvis_rot', pelvis_rot,
    #       'pelvis_vel', pelvis_vel,
    #       'left_hip_pos', left_hip_pos,
    #       'right_hip_pos', right_hip_pos,
    #       'joint_angles_arr', joint_angles_arr,
    #       'joint_rotations', joint_rotations,
    #       'joint_velocities', joint_velocities
    #       )

    motion_features = torch.cat([
        left_contact,
        right_contact,
        left_hip_rot,
        right_hip_rot,
        pelvis_rot,
        pelvis_vel,
        left_hip_pos,
        right_hip_pos,
        joint_angles_arr,
        joint_rotations,
        joint_velocities,
    ], dim=1)

    return motion_features


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack((w, x, y, z), -1)


def quaternion_inverse(q):
    q_inv = q.clone()
    q_inv[..., 1:] *= -1
    return q_inv / q.norm(dim=-1, keepdim=True).pow(2)


def rotate_vector(v, q):
    q_conj = quaternion_inverse(q)
    v_quat = torch.cat([torch.zeros(v.shape[0], 1), v], dim=-1)
    rotated_v = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return rotated_v[:, 1:]


def reconstruct_joints_from_features(motion_features):
    # Extract features from motion_features
    left_contact = motion_features[:, 0]
    right_contact = motion_features[:, 1]
    left_hip_rot = motion_features[:, 2:6]
    right_hip_rot = motion_features[:, 6:10]
    pelvis_rot = motion_features[:, 10:14]
    pelvis_vel = motion_features[:, 14:17]
    left_hip_pos = motion_features[:, 17:20]
    right_hip_pos = motion_features[:, 20:23]
    joint_angles_arr = motion_features[:, 23:150].view(-1, 127)
    joint_rotations = motion_features[:, 150:658].view(-1, 127, 4)
    joint_velocities = motion_features[:, 658:].view(-1, 127, 3)

    # Reconstruct joints from features
    root_positions = torch.cumsum(pelvis_vel, dim=0)
    root_rotations = pelvis_rot
    joint_positions = torch.zeros(root_positions.shape[0], 127, 3)

    joint_positions[:, 0] = root_positions
    joint_positions[:, 1] = root_positions + rotate_vector(left_hip_pos, root_rotations)
    joint_positions[:, 2] = root_positions + rotate_vector(right_hip_pos, root_rotations)

    for i in range(3, 127):
        parent_rot = joint_rotations[:, i - 1, :]
        parent_pos = joint_positions[:, i - 1, :]
        joint_angle = joint_angles_arr[:, i]
        local_pos = joint_velocities[:, i, :] * torch.sin(joint_angle)[:, None] / torch.sin(joint_angle - 0.5 * np.pi)[:, None]
        local_pos = torch.where(torch.isnan(local_pos), torch.zeros_like(local_pos), local_pos)
        global_pos = parent_pos + rotate_vector(local_pos, parent_rot)
        joint_positions[:, i] = global_pos

    return joint_positions


if __name__ == "__main__":
    # Convert 'joints' variable from your code to a torch tensor
    '''
    # pip install torchgeometry
    python -m process.motion_feat
    '''
    import numpy as np
    np.random.seed(0)
    joints = np.random.rand(10, 127, 3)
    joints = torch.tensor(joints, dtype=torch.float32)

    motion_features = extract_motion_features(joints)
    # print(motion_features.shape)

    # Reconstruct joints from features
    reconstructed_joints = reconstruct_joints_from_features(motion_features)
    # print(reconstructed_joints.shape)

    # Check if reconstructed joints are close to original joints
    print(joints, reconstructed_joints)
    print(torch.allclose(joints, reconstructed_joints, atol=1e-1))
