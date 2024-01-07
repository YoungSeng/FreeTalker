import scipy.ndimage.filters as filters
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb


def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()

# PyTorch-backed implementations
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qinv_np(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return qinv(torch.from_numpy(q).float()).numpy()

def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # pdb.set_trace()     # np.isnan(quaternions).any(), np.isinf(np.array(quaternions)).any()
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions + 1e-6).sum(-1)        # 20230823

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )       # np.isnan(np.array(o)).any(), np.isnan(np.array(two_s)).any()
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints





def main():
    import numpy as np

    smplx_joint_names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye",
        "right_eye",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3"
    ]

    # Define the kinematic tree for SMPLX
    smplx_kinematic_tree = [
        [0, 2, 5, 8, 11],  # 右侧下肢
        [0, 1, 4, 7, 10],  # 左侧下肢
        [0, 3, 6, 9, 12, 15],  # 脊柱和头部
        [9, 14, 17, 19, 21],  # 右侧上肢
        [9, 13, 16, 18, 20],  # 左侧上肢
        [15, 22],  # 颌部
        [15, 23],  # 左眼
        [15, 24],  # 右眼
        [20, 25, 26, 27],  # 左手食指
        [20, 28, 29, 30],  # 左手中指
        [20, 31, 32, 33],  # 左手小指
        [20, 34, 35, 36],  # 左手无名指
        [20, 37, 38, 39],  # 左手拇指
        [21, 40, 41, 42],  # 右手食指
        [21, 43, 44, 45],  # 右手中指
        [21, 46, 47, 48],  # 右手小指
        [21, 49, 50, 51],  # 右手无名指
        [21, 52, 53, 54],  # 右手拇指
    ]

    # Generate the offsets for SMPLX joints
    smplx_raw_offsets = np.zeros((len(smplx_joint_names), 3))
    for chain in smplx_kinematic_tree:
        for i in range(1, len(chain)):
            if chain[i - 1] < chain[i]:
                smplx_raw_offsets[chain[i]] = np.array([1, 0, 0])
            elif chain[i - 1] > chain[i]:
                smplx_raw_offsets[chain[i]] = np.array([-1, 0, 0])
            else:
                smplx_raw_offsets[chain[i]] = np.array([0, 1, 0])

    # Now you can use the `smplx_raw_offsets` in your Skeleton class
    return smplx_raw_offsets

if __name__ == '__main__':
    '''
    python process/skeleton.py
    '''
    smplx_raw_offsets = main()
    print(smplx_raw_offsets)
