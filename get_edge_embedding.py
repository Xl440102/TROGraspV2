import json, os
import torch
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from utils.rotation import matrix_to_vector
from utils.tro_hand_model import create_hand_model

def parse_urdf_topology(urdf_path):
 
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_to_joint_map = {}

    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        if "virtual" in joint_name: continue
        
        child_link = joint.find('child').get('link')
        parent_link = joint.find('parent').get('link')
        joint_type = joint.get('type')
        
        # Get Origin
        origin = joint.find('origin')
        xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
        rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
        
        # Get Axis
        axis_elem = joint.find('axis')
        axis = [float(x) for x in axis_elem.get('xyz', '1 0 0').split()] if axis_elem is not None else [0, 0, 1]
        
        # 3. Get Limits
        limit = joint.find('limit')
        if limit is not None:
            lower = float(limit.get('lower', 0))
            upper = float(limit.get('upper', 0))
        else:
            lower, upper = 0.0, 0.0  # fixed joints
            
        link_to_joint_map[child_link] = {
            'name': joint_name,
            'type': joint_type,
            'parent': parent_link,
            'xyz': xyz,
            'rpy': rpy,
            'axis': axis,
            'limits': (lower, upper)
        }
    return link_to_joint_map

def get_bps_embeddings(
    embodiments,
    R_max=0.15,
    random_seed=42,
    link_embed_dim=128
):

    # get BPS points, random seed fixed
    g = torch.Generator()
    g.manual_seed(random_seed)
    basis_points = torch.randn(link_embed_dim, 3, generator=g)
    basis_points = torch.nn.functional.normalize(basis_points, dim=1)
    random_radii = torch.rand(link_embed_dim, 1, generator=g).pow(1/3) * R_max
    basis_points = basis_points * random_radii

    # get feature
    bps_embedding_dict = {}
    for hand_name in tqdm(embodiments, desc="Generate link embed..."):

        hand_model = create_hand_model(hand_name)
        hand_bps_embedding = []
        
        for link_name in hand_model.link_names:
            ## BPS feature in link local frame
            link_vertices = torch.tensor(hand_model.vertices[link_name], dtype=torch.float32)
            pk_link = hand_model.pk_chain.find_link(link_name)
            T_offset = pk_link.visuals[0].offset.get_matrix()[0]
        
            homo_vertices = torch.cat([
                link_vertices, 
                torch.ones(link_vertices.shape[0], 1, device=link_vertices.device)
            ], dim=-1)
            link_vertices_local_frame = (homo_vertices @ T_offset.T)[:, :3]

            dists = torch.cdist(basis_points, link_vertices_local_frame) 
            link_bps_feature = dists.min(dim=1)[0]
            hand_bps_embedding.append(link_bps_feature)
        
        bps_embedding_dict[hand_name] = torch.stack(hand_bps_embedding, dim=0)
    return bps_embedding_dict


def get_se3_matrix(xyz, rpy, axis, theta):
 
    # Static Transform
    rot_base = R.from_euler('xyz', rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot_base
    T[:3, 3] = xyz
    
    # Rotation around axis by theta
    axis = np.array(axis)
    rot_motion = R.from_rotvec(axis * theta).as_matrix()
    T_motion = np.eye(4)
    T_motion[:3, :3] = rot_motion
    
    return torch.from_numpy(T @ T_motion).float()


def get_edge_embeddings(hand_names, joint_val_num=128):

    edge_embeddings_dict = {}
    for hand_name in tqdm(hand_names, desc="Generate edge embed..."):

        hand_model = create_hand_model(hand_name)
        urdf_path = urdf_path_dict["urdf_path"][hand_name]
        link_to_joint_map = parse_urdf_topology(urdf_path)

        num_link = len(hand_model.link_names)
        edge_embed = torch.zeros((num_link, num_link, joint_val_num, 6))

        for child_link, joint_info in link_to_joint_map.items():
            
            parent_link = joint_info["parent"]
            if child_link not in hand_model.link_names or parent_link not in hand_model.link_names:
                continue

            child_idx = hand_model.link_names.index(child_link)
            parent_idx = hand_model.link_names.index(parent_link)

            lower, upper = joint_info['limits']
            joint_spaces = np.linspace(lower, upper, num=joint_val_num)
            
            joint_feat = []
            for joint_value in joint_spaces:
                T_rel = get_se3_matrix(joint_info['xyz'], joint_info['rpy'], joint_info['axis'], joint_value)
                joint_feat.append(matrix_to_vector(T_rel))
            joint_feat = torch.stack(joint_feat, dim=0)
            
            edge_embed[child_idx, parent_idx] = joint_feat
            edge_embed[parent_idx, child_idx] = joint_feat
    
        edge_embeddings_dict[hand_name] = edge_embed
    return edge_embeddings_dict


def get_pos_embeddings(hand_names): 

    import viser
    import viser.transforms as tf
    server = viser.ViserServer()

    pos_embeddings_dict = {}
    for hand_name in tqdm(hand_names, desc="Get link pos in hand frame..."):

        server.scene.reset()
        hand_model = create_hand_model(hand_name)
        q = torch.zeros(hand_model.dof)

        robot_trimesh = hand_model.get_trimesh_q(q)['visual']
        server.scene.add_mesh_simple(
            hand_name,
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=1.0
        )
       
        all_link_frames = {}
        link_se3 = hand_model.get_link_se3(q)
        for link_name, se3 in zip(hand_model.link_names, link_se3):
            all_link_frames[link_name] = se3.cpu().numpy()

        for link_name, frame in all_link_frames.items():
            server.scene.add_frame(
                name=f"{hand_name}_{link_name}",
                wxyz=tf.SO3.from_matrix(frame[:3, :3]).wxyz,
                position=frame[:3, 3],
                axes_length=0.03,
                axes_radius=0.002
            )

        link_vecs = matrix_to_vector(link_se3)
        pos_embeddings_dict[hand_name] = torch.tensor(link_vecs, dtype=torch.float32)
        breakpoint()  # Manual Visual Check for Each Hand

    return pos_embeddings_dict
        
if __name__ == '__main__':

    hand_names = [
        "allegro_hand",
        "ability_hand",
        "barrett_hand",
        "dclaw_gripper",
        "leaphand",
        "shadow_hand",
        "schunk_hand",
        "xhand"
    ]
    urdf_path_dict = json.load(open("data/data_urdf/robot/urdf_assets_meta.json"))
    
    # Get BPS link embeddings
    bps_embeddings_dict = get_bps_embeddings(embodiments=hand_names)
    for hand_name, bps_embedding in bps_embeddings_dict.items():
        os.makedirs(f"data/data_urdf/robot/{hand_name}/feat", exist_ok=True)
        torch.save(bps_embedding, f"data/data_urdf/robot/{hand_name}/feat/bps.pt")

    # Get edge embeddings
    link_embeddings_dict = get_edge_embeddings(hand_names)
    for hand_name, edge_embedding in link_embeddings_dict.items():
        torch.save(edge_embedding, f"data/data_urdf/robot/{hand_name}/feat/edge.pt")

    # Get pos embeddings
    pos_embeddings_dict = get_pos_embeddings(hand_names)
    for hand_name, pos_embedding in pos_embeddings_dict.items():
        torch.save(pos_embedding, f"data/data_urdf/robot/{hand_name}/feat/pos.pt")