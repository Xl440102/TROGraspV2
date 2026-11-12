#################################
# T(R,O,E) in score matching scheme
#################################
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3

from utils.rotation import *
from model.tro_denoiser import TROEGraphDenoiser
from utils.tro_hand_model import create_hand_model


class LinkPosEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        out_dim=128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, link_pose):
        out = self.encoder(link_pose)
        return out

class JointSpaceEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        out_dim=128
    ):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, edge_embed):
        """
        edge_embed: [M, N, 6]
        return:     [M, F]
        """
        h = self.phi(edge_embed)           # [M, N, H]

        h_max = h.max(dim=-2).values       # [M, H]
        h_mean = h.mean(dim=-2)            # [M, H]

        h_set = torch.cat([h_max, h_mean], dim=-1)   # [M, 2H]
        out = self.rho(h_set)              # [M, F]
        return out
    

class RobotGraph(nn.Module):

    def __init__(
        self,
        object_patch,
        env_patch,
        max_link_node,
        link_embed_dim,
        edge_embed_dim,
        joint_feat_dim,
        link_pos_dim,
        robot_links,
        inference_config,
        N_t_training,
        diffusion_config,
        denoiser_config,
        embodiment,
        dtype,
        mode="train"
    ):

        super(RobotGraph, self).__init__()
        self.mode = mode
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        # meta
        self.embodiment = embodiment
        self.object_patch = object_patch
        self.env_patch = env_patch
        self.max_link_node = max_link_node
     
        self.N_t_training = N_t_training
        self.init_diffusion(diffusion_config)
        self.denoiser = TROEGraphDenoiser(
            object_patch=self.object_patch,
            env_patch=self.env_patch,
            max_link_node=self.max_link_node,
            **OmegaConf.to_container(denoiser_config, resolve=True)
        )

        self.mode = mode
        if self.mode == "train":
            self.is_mount = False   # The model is trained without mount by default
        elif self.mode == "test":
            inference_config = OmegaConf.to_container(inference_config, resolve=True)
            self.inference_step = inference_config["inference_step"]
            self.is_mount = inference_config["is_mount"]
            if self.is_mount:
                self.hand_links_pc = torch.load(inference_config["hand_pc_path"], weights_only=False)
        
        if self.is_mount:
            self.hand_dict = {}
            for hand_name in self.embodiment:
                self.hand_dict[hand_name] = create_hand_model(hand_name, has_mount=True)
        else:
            self.hand_dict = {}
            for hand_name in self.embodiment:
                self.hand_dict[hand_name] = create_hand_model(hand_name)

        # link embedding
        self.robot_links = robot_links
        self.link_embed_dim = link_embed_dim
        self.link_embeddings = {}
        for hand_name in self.embodiment:
            self.link_embeddings[hand_name] = torch.load(f"data/data_urdf/robot/{hand_name}/feat/bps.pt", weights_only=False)
    
        # edge embedding
        self.edge_embed_dim = edge_embed_dim
        self.joint_feat_dim = joint_feat_dim
        self.edge_embeddings = {}
        for hand_name in self.embodiment:
            self.edge_embeddings[hand_name] = torch.load(f"data/data_urdf/robot/{hand_name}/feat/edge.pt", weights_only=False)
        self.joint_space_encoder = JointSpaceEncoder(
            hidden_dim=self.joint_feat_dim, out_dim=self.joint_feat_dim
        )

        # pos embedding
        self.link_pos_dim = link_pos_dim
        self.pos_embeddings = {}
        for hand_name in self.embodiment:
            self.pos_embeddings[hand_name] = torch.load(f"data/data_urdf/robot/{hand_name}/feat/pos.pt", weights_only=False)
        self.link_pos_encoder = LinkPosEncoder(
            hidden_dim=self.link_pos_dim, out_dim=self.link_pos_dim
        )
    

    def _unit_ball_(self, pc):

        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        max_radius = pc.norm(dim=-1).max()
        return centroid, max_radius

    def init_diffusion(self, cfg):
        self.M = cfg["M"]
        self.scheduling = cfg["scheduling"]

        if self.scheduling == "linear":
            self.beta_min = cfg["beta_min"]
            self.beta_max = cfg["beta_max"]
            betas = torch.linspace(
                self.beta_min,
                self.beta_max,
                self.M,
                dtype=torch.float32
            )

        elif self.scheduling == "cosine":
            s = cfg["cosine_s"]

            steps = self.M + 1
            x = torch.linspace(0, self.M, steps, dtype=torch.float32)

            alpha_bar = torch.cos(
                ((x / self.M) + s) / (1 + s) * torch.pi / 2
            ) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]

            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(min=1e-5, max=0.999)

        else:
            raise NotImplementedError(f"Unknown scheduling: {self.scheduling}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        self.eta = cfg["ddim_eta"]
        self.noise_lambda = cfg["lambda"]


    def _expand_and_reshape_(self, x, name):

        shape = x.shape
        B = x.shape[0]
        if len(shape) == 3:  # Node
            return (
                x[:, None, :, :]
                .expand(-1, self.N_t_training, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2])
            )
        elif len(shape) == 4:  # Edge
            return (
                x[:, None, :, :, :]
                .expand(-1, self.N_t_training, -1, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2], shape[3])
            )
        elif len(shape) == 2:  # Node Mask
            return (
                x[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(B * self.N_t_training, shape[1])
            )
        else:
            raise ValueError(f"Unsupported shape for {name}: {shape}")

    def forward(self, batch, eps=1e-12):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device

        ## Graph Construction
        env_centroids = batch["env_centroids"]
        env_scale = batch["env_scale"]

        object_nodes = torch.cat([
            batch["object_xyz"],
            batch["object_feat"]
        ], dim=-1)  # [B, P, 3+64]

        env_nodes = torch.cat([
            batch["env_xyz"],
            batch["env_feat"]
        ], dim=-1)  # [B, E, 3+64]

        # Target Link Node
        link_target_poses = batch["target_vec"]
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=self.dtype
        )
        link_pos_embeds = torch.zeros(
            [B, self.max_link_node, self.link_pos_dim], device=device, dtype=self.dtype
        )
        link_node_masks = torch.zeros(
            [B, self.max_link_node], device=device, dtype=torch.bool
        )
        joint_space_se3 = torch.zeros(
            [B, self.max_link_node, self.max_link_node, self.edge_embed_dim, 6], device=device, dtype=self.dtype
        )  # [B, L, L, 128, 6]

        for batch_id in range(B):
            robot_name = self.embodiment[int(batch["robot_id"][batch_id].item())]
            num_link = self.robot_links[robot_name]
            link_robot_embeds[batch_id, :num_link, :] = self.link_embeddings[robot_name].to(device)
            link_pos_embeds[batch_id, :num_link, :] = self.link_pos_encoder(self.pos_embeddings[robot_name].to(device))
            link_node_masks[batch_id, :num_link] = True
            joint_space_se3[batch_id, :num_link, :num_link] = self.edge_embeddings[robot_name].to(device)
        
        robot_nodes = torch.cat([
            link_target_poses, 
            link_robot_embeds,
            link_pos_embeds
        ], dim=-1)  # [B, L, 6+128+128]
        joint_space_feat = self.joint_space_encoder(joint_space_se3)  # [B, L, L, F]
    
        ## Forward Process
        t = np.random.randint(0, self.M, (B * self.N_t_training))
        V_O = self._expand_and_reshape_(object_nodes, "V_O")             # [B*T, P, 68]
        V_E = self._expand_and_reshape_(env_nodes, "V_E")                # [B*T, E, 68]
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")              # [B*T, L, 280]
        F_J = self._expand_and_reshape_(joint_space_feat, "F_J")         # [B*T, L, L, F]

        centroids = self._expand_and_reshape_(env_centroids, "center")       # [B*T, 1, 3]
        scale = self._expand_and_reshape_(env_scale, "scale")                # [B*T, 1, 1]
        V_R_trans, V_R_rot, V_R_embed = V_R[:, :, :3], V_R[:, :, 3:6], V_R[:, :, 6:]
        V_R_trans = (V_R_trans - centroids) / scale

        eta_V_R_trans = torch.randn_like(V_R_trans)
        eta_V_R_rot = torch.randn_like(V_R_rot)
        a_bar = self.alpha_bars[t][:, None, None]
        
        noisy_trans = a_bar.sqrt() * V_R_trans + (1 - a_bar).sqrt() * eta_V_R_trans
        noisy_rot = a_bar.sqrt() * V_R_rot + (1 - a_bar).sqrt() * eta_V_R_rot
        noisy_V_R = torch.cat([noisy_trans, noisy_rot, V_R_embed], dim=-1)

        # update graph edges
        noisy_V_R_pose = noisy_V_R[:, :, :6]
        noisy_E_RR = relative_pose_6d(noisy_V_R_pose, noisy_V_R_pose)
        noisy_E_RR = torch.cat([noisy_E_RR, F_J], dim=-1)

        object_positions = V_O[:, :, :3]
        B, P, _ = object_positions.shape
        object_rots = torch.zeros((B, P, 3), device=device, dtype=object_positions.dtype)
        object_6D_pose = torch.cat([object_positions, object_rots], dim=-1)
        noisy_E_OR = relative_pose_6d(noisy_V_R_pose, object_6D_pose)

        env_positions = V_E[:, :, :3]
        B, E, _ = env_positions.shape
        env_rots = torch.zeros((B, E, 3), device=device, dtype=env_positions.dtype)
        env_6D_pose = torch.cat([env_positions, env_rots], dim=-1)
        noisy_E_ER = relative_pose_6d(noisy_V_R_pose, env_6D_pose)

        ## Backward Denoising  
        pred_noise = self.denoiser(
            V_O,
            noisy_V_R,
            V_E,
            noisy_E_OR,
            noisy_E_RR,
            noisy_E_ER,
            t
        )
        
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_trans_noise = pred_noise[:, :, :3]
        pred_rot_noise = pred_noise[:, :, 3:]

        err_trans_noise = ((eta_V_R_trans - pred_trans_noise) ** 2).mean(dim=-1)
        loss_trans_noise = (err_trans_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        err_rot_noise = ((eta_V_R_rot - pred_rot_noise) ** 2).mean(dim=-1)
        loss_rot_noise = (err_rot_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        total_loss = loss_trans_noise + loss_rot_noise
        loss_dict = {
            "loss_rot": loss_rot_noise,
            "loss_trans": loss_trans_noise,
            "loss_total": total_loss
        }
        return loss_dict
    
    def project_pose_with_env_pc(
        self,
        x_prev_pose_real,
        env_pc,
        robot_name,
        num_iters=5,
        margin=0.01,
        lambda_coll=25.0,
        lr=0.1
    ):
        hand = self.hand_dict[robot_name]

        with torch.enable_grad():
            x = x_prev_pose_real.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=lr)

            for it in range(num_iters):
                optimizer.zero_grad()

                T = vector_to_matrix(x)[:, :self.robot_links[robot_name]]
                links_pc = hand.get_transformed_links_pc_from_se3(
                    T, links_pc=self.hand_links_pc
                )

                dists = torch.cdist(links_pc, env_pc)   # [B, N_link_pts, N_env]
                min_dist, _ = dists.min(dim=-1)         # [B, N_link_pts]

                penetration = torch.relu(margin - min_dist)
                collision_loss = (penetration ** 2).mean()

                pose_deviation = ((x - x_prev_pose_real) ** 2).mean()
                proj_loss = pose_deviation + lambda_coll * collision_loss

                proj_loss.backward()
                optimizer.step()

            return x.detach()

    @staticmethod
    def _make_noise_bank(noise_seeds, num_draws, shape, device, dtype):
        """Per-candidate Gaussian noise, reproducible from one integer seed each.

        Every candidate draws all the noise it will use (initial noise + per-step
        DDIM noise) from its own CPU generator in a single call, so a candidate's
        sample depends only on its seed, not on batch size, chunking or order.
        Returns [B, num_draws, *shape].
        """
        bank = []
        for seed in noise_seeds:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            bank.append(torch.randn((num_draws, *shape), generator=g, dtype=torch.float32))
        return torch.stack(bank, dim=0).to(device=device, dtype=dtype)

    def inference(self, batch, eps=1e-8, noise_seeds=None):
        """Reverse DDIM sampling.

        noise_seeds: optional list of B ints. If given, all random noise for
        candidate b comes from seed noise_seeds[b] (see _make_noise_bank);
        if None, the original global-RNG behaviour is kept.
        """

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        robot_name = self.embodiment[int(batch["robot_id"][0].item())]
        num_link = self.robot_links[robot_name]
        link_names = self.hand_dict[robot_name].link_names

        env_centroids = batch["env_centroids"]
        env_scale = batch["env_scale"]
            
        object_nodes = torch.cat([
            batch["object_xyz"],
            batch["object_feat"]
        ], dim=-1)  # [B, P, 3+64]

        env_nodes = torch.cat([
            batch["env_xyz"],
            batch["env_feat"]
        ], dim=-1)  # [B, E, 3+64]

        ## DDIM schedule (computed first so the noise bank knows how many draws)
        step = self.M // self.inference_step
        start_t = self.M - 1 - step
        ddim_t = torch.arange(start_t, -1, -step, device=device, dtype=torch.long)

        noise_bank = None
        if noise_seeds is not None:
            assert len(noise_seeds) == B, f"noise_seeds has {len(noise_seeds)} entries, batch has {B}"
            # draws: [init_trans, init_rot, (z_trans, z_rot) per step]
            noise_bank = self._make_noise_bank(
                noise_seeds, 2 + 2 * len(ddim_t), (self.max_link_node, 3), device, dtype
            )

        ## Start from complete noise of link node (pose)
        if noise_bank is None:
            noisy_V_R_trans = torch.randn(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            noisy_V_R_rot = torch.randn(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
        else:
            noisy_V_R_trans = noise_bank[:, 0]
            noisy_V_R_rot = noise_bank[:, 1]
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        link_pos_embeds = torch.zeros(
            [B, self.max_link_node, self.link_pos_dim], device=device, dtype=self.dtype
        )
        joint_space_se3 = torch.zeros(
            [B, self.max_link_node, self.max_link_node, self.edge_embed_dim, 6], device=device, dtype=self.dtype
        )  # [B, L, L, 128, 6]

        link_embed = self.link_embeddings[robot_name].to(device)
        link_robot_embeds[:, :num_link, :] = link_embed.unsqueeze(0).expand(B, -1, -1)
        link_pos_embed = self.link_pos_encoder(self.pos_embeddings[robot_name].to(device))
        link_pos_embeds[:, :num_link, :] = link_pos_embed.unsqueeze(0).expand(B, -1, -1)
        noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds, link_pos_embeds], dim=-1)

        joint_space_se3[:, :num_link, :num_link] = self.edge_embeddings[robot_name].to(device).unsqueeze(0).expand(B, -1, -1, -1, -1)
        joint_space_feat = self.joint_space_encoder(joint_space_se3)  # [B, L, L, F]

        ## Formulate edges
        noisy_V_R_pose = noisy_V_R[:, :, :6]
        noisy_E_RR = relative_pose_6d(noisy_V_R_pose, noisy_V_R_pose)
        noisy_E_RR = torch.cat([noisy_E_RR, joint_space_feat], dim=-1)

        object_positions = object_nodes[:, :, :3]
        B, P, _ = object_positions.shape
        object_rots = torch.zeros((B, P, 3), device=device, dtype=object_positions.dtype)
        object_6D_pose = torch.cat([object_positions, object_rots], dim=-1)
        noisy_E_OR = relative_pose_6d(noisy_V_R_pose, object_6D_pose)

        env_positions = env_nodes[:, :, :3]
        B, E, _ = env_positions.shape
        env_rots = torch.zeros((B, E, 3), device=device, dtype=env_positions.dtype)
        env_6D_pose = torch.cat([env_positions, env_rots], dim=-1)
        noisy_E_ER = relative_pose_6d(noisy_V_R_pose, env_6D_pose)

        ## Reverse DDIM
        predict_link_pose_list = []

        for i, diffuse_step in enumerate(ddim_t):
            
            diffuse_step = diffuse_step.item()

            # predict noise
            pred_noise_world = self.denoiser(
                object_nodes,
                noisy_V_R,
                env_nodes,
                noisy_E_OR,
                noisy_E_RR,
                noisy_E_ER,
                t=torch.full((B,), diffuse_step, dtype=torch.long, device=device)
            )

            trans_noise = pred_noise_world[:, :, :3]
            rot_noise = pred_noise_world[:, :, 3:]

            # predict x_0
            a_bar_t = self.alpha_bars[diffuse_step]
            if i == len(ddim_t) - 1:
                a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                a_bar_prev = self.alpha_bars[ddim_t[i + 1]]
            
            x_t_trans = noisy_V_R_trans
            x_t_rot = noisy_V_R_rot

            x_0_trans = (x_t_trans - (1 - a_bar_t).sqrt() * trans_noise) / a_bar_t.sqrt()
            x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * rot_noise) / a_bar_t.sqrt()

            sigma_t = self.eta * torch.sqrt(((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev))            
            ddim_coeff = torch.sqrt(1 - a_bar_prev - sigma_t ** 2)

            if i == len(ddim_t) - 1:
                z_trans = torch.zeros_like(x_0_trans)
                z_rot = torch.zeros_like(x_0_rot)
            elif noise_bank is None:
                z_trans = torch.randn_like(x_0_trans)
                z_rot = torch.randn_like(x_0_rot)
            else:
                z_trans = noise_bank[:, 2 + 2 * i].to(x_0_trans.dtype)
                z_rot = noise_bank[:, 3 + 2 * i].to(x_0_rot.dtype)

            noisy_V_R_trans = a_bar_prev.sqrt() * x_0_trans + ddim_coeff * trans_noise + sigma_t * z_trans * self.noise_lambda
            noisy_V_R_rot   = a_bar_prev.sqrt() * x_0_rot   + ddim_coeff * rot_noise   + sigma_t * z_rot   * self.noise_lambda
    
            # update node and edge
            noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds, link_pos_embeds], dim=-1)
            noisy_V_R_pose = noisy_V_R[:, :, :6]

            noisy_E_RR = relative_pose_6d(noisy_V_R_pose, noisy_V_R_pose)
            noisy_E_RR = torch.cat([noisy_E_RR, joint_space_feat], dim=-1)
            noisy_E_OR = relative_pose_6d(noisy_V_R_pose, object_6D_pose)
            noisy_E_ER = relative_pose_6d(noisy_V_R_pose, env_6D_pose)
            
            pred_pose = noisy_V_R_pose.clone()
            pred_pose[:, :, :3] = pred_pose[:, :, :3] * env_scale + env_centroids
            predict_link_pose_dict = {}
            predict_link_pose = vector_to_matrix(pred_pose[:, :num_link])
 
            for link_id, link_name in enumerate(link_names):
                predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
            predict_link_pose_list.append(predict_link_pose_dict)

        return predict_link_pose_list