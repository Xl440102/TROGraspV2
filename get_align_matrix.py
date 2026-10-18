import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.tro_hand_model import create_hand_model

def get_se3_from_trans_and_rpy(trans, rpy):
        
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = trans
    R_mat = R.from_euler('XYZ', rpy).as_matrix()
    T[:3, :3] = R_mat
    return T


if __name__ == '__main__':

    hand_types = ['allegro_hand', 'barrett_hand', 'shadow_hand', 'leaphand', 'xhand', 'ability_hand', 'schunk_hand', 'dclaw_gripper']
    T_base_palm_dict = {}

    for hand_type in hand_types:

        if hand_type == 'shadow_hand':
            T_cylinder_forearm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            T_forearm_wrist = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, -0.010, 0.21301], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            T_wrist_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.034], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            T_cylinder_palm = T_cylinder_forearm @ T_forearm_wrist @ T_wrist_palm
            T_base_palm = T_cylinder_palm

        elif hand_type == 'allegro_hand':
            T_cylinder_palm = get_se3_from_trans_and_rpy(
                trans=np.array([-0.04, 0.0, 0.18], dtype=np.float32),
                rpy=np.array([0.0, -1.5707, 0.0], dtype=np.float32)
            )
            T_base_palm = T_cylinder_palm

        elif hand_type == 'barrett_hand':
            T_cylinder_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.15], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            T_base_palm = T_cylinder_palm

        elif hand_type == 'leaphand':
            T_base_cylinder = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.038, 0.098], dtype=np.float32),
                rpy=np.array([0.0, -1.57079, 0.0], dtype=np.float32)
            )
            T_cylinder_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.07, 0.03, 0.0], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            T_base_palm = T_base_cylinder @ T_cylinder_palm

        elif hand_type == 'xhand':
            T_base_cylinder = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                rpy=np.array([3.1415926, 0.0, 1.5707963], dtype=np.float32)
            )
            T_cylinder_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.15], dtype=np.float32),
                rpy=np.array([3.1415926, 0.0, 0.0], dtype=np.float32)
            )
            T_base_palm = T_base_cylinder @ T_cylinder_palm

        elif hand_type == 'ability_hand':
            T_base_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.15], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )

        elif hand_type == 'schunk_hand':
            T_base_cylinder = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                rpy=np.array([0.0, 0.0, -1.57079], dtype=np.float32)
            )
            T_cylinder_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, -0.01, 0.15], dtype=np.float32),
                rpy=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            T_base_palm = T_base_cylinder @ T_cylinder_palm

        elif hand_type == 'dclaw_gripper':
            T_base_cylinder = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                rpy=np.array([0.0, -1.57079, 0.0], dtype=np.float32)
            )
            T_cylinder_palm = get_se3_from_trans_and_rpy(
                trans=np.array([0.0, 0.0, 0.15], dtype=np.float32),
                rpy=np.array([0.0, -1.57079, 0.0], dtype=np.float32)
            )
            T_base_palm = T_base_cylinder @ T_cylinder_palm

        T_base_palm_dict[hand_type] = T_base_palm

    torch.save(T_base_palm_dict, "/home/fx/workspace/TRO/data/data_urdf/robot/align_ft_shadow.pt")