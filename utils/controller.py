import os
import sys
import time
import json
import trimesh
import torch
import viser

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.tro_hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler


def get_link_dir(robot_name, joint_name):
    if joint_name.startswith('virtual'):
        return None

    if robot_name == 'allegro_hand':
        if joint_name in ['joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_13.0']:
            return None
        link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)

    elif robot_name == 'barrett_hand':
        if joint_name in ['bh_j11_joint', 'bh_j21_joint']:
            return None
        link_dir = torch.tensor([-1, 0, 0], dtype=torch.float32)

    elif robot_name == 'wuji_hand':
        if joint_name in [
            "finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4",
            "finger2_joint1", "finger2_joint2", "finger2_joint3", "finger2_joint4",
            "finger3_joint1", "finger3_joint2", "finger3_joint3", "finger3_joint4",
            "finger4_joint1", "finger4_joint2", "finger4_joint3", "finger4_joint4",
            "finger5_joint1", "finger5_joint2", "finger5_joint3", "finger5_joint4",
        ]:
            link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
        else:
            return None

    elif robot_name == 'shadow_hand':
        if joint_name in ['WRJ2', 'WRJ1']:
            return None
        if joint_name != 'THJ5':
            link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
        else:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

    elif robot_name == 'leaphand':
        if joint_name in ['13']:
            return None
        if joint_name in ['0', '4', '8']:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
        elif joint_name in ['1', '5', '9', '12', '14']:
            link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
        else:
            link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)

    elif robot_name == "xhand":
        # thumb
        if joint_name in [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        # four fingers
        elif joint_name in [
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",

            "right_hand_mid_joint1",
            "right_hand_mid_joint2",

            "right_hand_ring_joint1",
            "right_hand_ring_joint2",

            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]:
            link_dir = torch.tensor([0, 0, -1], dtype=torch.float32)

        else:
            return None
        
    elif robot_name == "xhand_left":
        # thumb
        if joint_name in [
            "left_hand_thumb_bend_joint",
            "left_hand_thumb_rota_joint1",
            "left_hand_thumb_rota_joint2",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        # four fingers
        elif joint_name in [
            "left_hand_index_bend_joint",
            "left_hand_index_joint1",
            "left_hand_index_joint2",

            "left_hand_mid_joint1",
            "left_hand_mid_joint2",

            "left_hand_ring_joint1",
            "left_hand_ring_joint2",

            "left_hand_pinky_joint1",
            "left_hand_pinky_joint2",
        ]:
            link_dir = torch.tensor([0, 0, -1], dtype=torch.float32)

        else:
            return None
        
    elif robot_name == "ability_hand":
        if joint_name in [
            "thumb_q1", "thumb_q2",
            "index_q1", "index_q2",
            "middle_q1", "middle_q2",
            "ring_q1", "ring_q2",
            "pinky_q1", "pinky_q2",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
        else:
            return None
        
    elif robot_name == "schunk_hand":
        if joint_name in [
            "right_hand_j5",
            "right_hand_index_spread",
            "right_hand_ring_spread",
            "right_hand_Finger_Spread",
        ]:
            return None

        # thumb chain
        elif joint_name in [
            "right_hand_Thumb_Opposition",
            "right_hand_Thumb_Flexion",
            "right_hand_j3",
            "right_hand_j4",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        # index finger
        elif joint_name in [
            "right_hand_Index_Finger_Proximal",
            "right_hand_Index_Finger_Distal",
            "right_hand_j14",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        # middle finger
        elif joint_name in [
            "right_hand_Middle_Finger_Proximal",
            "right_hand_Middle_Finger_Distal",
            "right_hand_j15",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        # ring finger
        elif joint_name in [
            "right_hand_Ring_Finger",
            "right_hand_j12",
            "right_hand_j16",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        # pinky finger
        elif joint_name in [
            "right_hand_Pinky",
            "right_hand_j13",
            "right_hand_j17",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)

        else:
            return None
    elif robot_name == "dclaw_gripper":
        if joint_name in [
            "joint_f1_0", "joint_f1_1", "joint_f1_2",
            "joint_f2_0", "joint_f2_1", "joint_f2_2",
            "joint_f3_0", "joint_f3_1", "joint_f3_2",
        ]:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
        else:
            return None
    else:
        raise NotImplementedError(f"Unknown robot name: {robot_name}!")

    return link_dir


def controller(robot_name, q_para):
    q_batch = torch.atleast_2d(q_para)

    hand = create_hand_model(robot_name, device=q_batch.device)
    joint_orders = hand.get_joint_orders()
    pk_chain = hand.pk_chain
    if q_batch.shape[-1] != len(pk_chain.get_joint_parameter_names()):
        q_batch = q_rot6d_to_q_euler(q_batch)
    status = pk_chain.forward_kinematics(q_batch)

    outer_q_batch = []
    inner_q_batch = []
    for batch_idx in range(q_batch.shape[0]):
        joint_dots = {}
        for frame_name in pk_chain.get_frame_names():
            frame = pk_chain.find_frame(frame_name)
            joint = frame.joint
            link_dir = get_link_dir(robot_name, joint.name)
            if link_dir is None:
                continue

            frame_transform = status[frame_name].get_matrix()[batch_idx]
            axis_dir = frame_transform[:3, :3] @ joint.axis
            link_dir = frame_transform[:3, :3] @ link_dir
            normal_dir = torch.cross(axis_dir, link_dir, dim=0)
            axis_origin = frame_transform[:3, 3]
            origin_dir = -axis_origin / torch.norm(axis_origin)
            joint_dots[joint.name] = torch.dot(normal_dir, origin_dir)

        q = q_batch[batch_idx]
        lower_q, upper_q = hand.pk_chain.get_joint_limits()
        outer_q, inner_q = q.clone(), q.clone()
  
        for joint_name, dot in joint_dots.items():
            idx = joint_orders.index(joint_name)
            if robot_name == 'robotiq_3finger':  # open -> upper, close -> lower
                outer_q[idx] += 0.25 * ((outer_q[idx] - lower_q[idx]) if dot <= 0 else (outer_q[idx] - upper_q[idx]))
                inner_q[idx] += 0.15 * ((inner_q[idx] - upper_q[idx]) if dot <= 0 else (inner_q[idx] - lower_q[idx]))
            else:  # open -> lower, close -> upper
                outer_q[idx] += 0.25 * ((lower_q[idx] - outer_q[idx]) if dot >= 0 else (upper_q[idx] - outer_q[idx]))
                inner_q[idx] += 0.15 * ((upper_q[idx] - inner_q[idx]) if dot >= 0 else (lower_q[idx] - inner_q[idx]))
        outer_q_batch.append(outer_q)
        inner_q_batch.append(inner_q)

    outer_q_batch = torch.stack(outer_q_batch, dim=0)
    inner_q_batch = torch.stack(inner_q_batch, dim=0)

    if q_para.ndim == 2:  # batch
        return outer_q_batch.to(q_para.device), inner_q_batch.to(q_para.device)
    else:
        return outer_q_batch[0].to(q_para.device), inner_q_batch[0].to(q_para.device)
