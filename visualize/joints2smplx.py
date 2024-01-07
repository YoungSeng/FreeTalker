# import numpy as np
# import torch
# from smplx import SMPLX
# from smplx.lbs import vertices2joints
#
# import sys
# [sys.path.append(i) for i in ['../smplify-x', '../smplify-x/smplifyx', '..']]
#
# from smplifyx.optimizers import JointOptimizer3D
# from smplifyx.prior import MaxMixturePrior
# from smplifyx.utils import to_tensor
#
#
#
#
# import torch
# from human_body_prior.body_model.body_model import BodyModel
# from human_body_prior.tools.visualization_tools import render_smpl_params
# from human_body_prior.tools.visualization_tools import imagearray2file
# from human_body_prior.tools.model_loader import load_vposer as load_vposer_orig
# from smplx.lbs import vertices2joints
#
# def load_vposer(vposer_ckpt, vp_model='snapshot'):
#     vposer_pt = torch.load(vposer_ckpt)
#     vposer = VPoser(vp_model=vp_model, **vposer_pt['model_params'])
#     vposer.load_state_dict(vposer_pt)
#     vposer.eval()
#     return vposer
#
# # Load the VPoser model
# vposer_ckpt = 'path/to/vposer.pth.tar'
# vposer = load_vposer(vposer_ckpt)
#
# # Set up the SMPLify-X optimizer
# joint_prior = MaxMixturePrior(prior_folder='path/to/smplx/models')
# joint_optimizer = JointOptimizer3D(model, joint_prior, vposer=vposer)
#
#
#
#
#
# # Load the joint positions data
# joint_positions = np.load('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/result_rec.npy')  # shape (len, 55, 3)
#
# # Map the given joint names to SMPL-X joint names
# smplx_joint_names = [
#     'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
#     'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
#     'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
#     'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw',
#     'left_eye', 'right_eye', 'left_index1', 'left_index2', 'left_index3',
#     'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2',
#     'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1',
#     'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3',
#     'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2',
#     'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1',
#     'right_thumb2', 'right_thumb3'
# ]
#
# # Create the SMPL-X model
# model = SMPLX(
#     model_path='/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models',
#     model_type='smplx',
#     gender='neutral',
#     ext='npz',
#     use_face_contour=False,
#     create_glb=False,
#     create_shape=True,
# )
#
# # Set up the SMPLify-X optimizer
# joint_prior = MaxMixturePrior(prior_folder='/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models')
# joint_optimizer = JointOptimizer3D(model, joint_prior)
#
# # Prepare the joint positions data
# input_joints = torch.tensor(joint_positions, dtype=torch.float32)
#
# # Optimize the SMPL-X parameters
# trans, poses = joint_optimizer(input_joints)
#
# # Save the optimized parameters as a npz file
# np.savez('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/result_rec.npy/smplx_output.npz',
#          trans=trans.detach().cpu().numpy(), poses=poses.detach().cpu().numpy())



'''
python -m visualize.joints2smplx
'''

import torch
import torch.optim as optim
import pdb
import numpy as np
import smplx
from smplx import SMPLX
from smplx.lbs import vertices2joints
import sys
[sys.path.append(i) for i in ['../smplify-x', '../smplify-x/smplifyx', '..', './process']]
from smplifyx.prior import MaxMixturePrior
from smplifyx.utils import to_tensor
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from process.motion_representation import recover_from_ric

import torch
import torch.optim as optim

class JointOptimizer3D:
    def __init__(self, model, joint_prior, vposer, num_iters=1000, learning_rate=2, lr_decay_rate=0.995):
        self.model = model
        self.joint_prior = joint_prior
        self.vposer = vposer
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate

    def __call__(self, input_joints):
        # Initialize the model parameters
        self.model.eval().cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        input_joints = input_joints.cuda()
        # Create learnable pose and translation parameters
        poses = torch.tensor(torch.zeros(input_joints.shape[0], 165), requires_grad=True, device='cuda')  # * 0.2 - 0.1
        trans = torch.tensor(torch.zeros(input_joints.shape[0], 3), requires_grad=True, device='cuda')  # * 0.2 - 0.1

        # Optimize the model parameters
        # optimizer = optim.AdamW([poses, trans], lr=self.learning_rate)
        optimizer = optim.Adam([poses, trans], lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay_rate)

        for i in range(self.num_iters):
            optimizer.zero_grad()
            output_joints = self.model(body_pose=poses[:, 3:66], global_orient=poses[:, :3], transl=trans[:, :3],
                       jaw_pose=poses[:, 66:69], leye_pose=poses[:, 69:72],
                       reye_pose=poses[:, 72:75], left_hand_pose=poses[:, 75:120],
                       right_hand_pose=poses[:, 120:165], return_verts=False).joints[:, :55]
            loss = torch.sum((output_joints - input_joints) ** 2)
            print('epoch: ', i, 'loss: ', loss.item(), 'lr: ', scheduler.get_last_lr()[0])
            # poch:  284 loss:  135.11245727539062 lr:  1.204272962881207
            # epoch:  999 loss:  45.90663146972656 lr:  0.01337481121373256
            # loss += self.joint_prior(torch.cat((poses[:, :66], trans[:, :3]), dim=-1), betas=None)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Return the optimized translation and poses
        return trans.detach().cpu(), poses.detach().cpu()


def export_to_obj(vertices, faces, output_file):
    with open(output_file, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


# Load the VPoser model
expr_dir = '/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/vposer_v2_05'
vposer, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)

# Load the joint positions data
joint_positions = np.load('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/result_rec.npy')  # shape (len, 55, 3)

# Create the SMPL-X model

model_path = "/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models"
model = smplx.create(model_path, model_type='smplx', gender='neutral', ext='npz', num_pca_comps=12,
                         create_global_orient=True, create_body_pose=True, create_betas=True,
                         create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                         create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                         use_face_contour=False, batch_size=360)
model.use_pca = False
# Set up the SMPLify-X optimizer
joint_prior = MaxMixturePrior(num_gaussians=8, prior_folder='/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads')
joint_optimizer = JointOptimizer3D(model, joint_prior, vposer=vposer)

# Prepare the joint positions data
input_joints = torch.tensor(joint_positions, dtype=torch.float32)

# Optimize the SMPL-X parameters
trans, poses = joint_optimizer(input_joints)

# Save the optimized parameters as a npz file
np.savez('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/smplx_output.npz',
         trans=trans.detach().cpu().numpy(), poses=poses.detach().cpu().numpy(), surface_model_type='smplx',
         gender='neutral', mocap_frame_rate=20, betas=np.zeros(16))


# # Get the vertices and faces of the optimized model
# with torch.no_grad():
#     vertices = model(betas=model.betas, body_pose=poses[:, 3:], global_orient=poses[:, :3], pose2rot=False, return_landmarks=False).vertices
#     faces = model.faces
#
# # Export the vertices and faces to an OBJ file
# export_to_obj(vertices.detach().cpu().numpy(), faces, '/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000400000/output.obj')


