import torch
import torch.nn as nn
from model.vqvae.vq_vae_ft import VQVAE

class VQVAEEncoder(nn.Module):

    def __init__(
        self, 
        vqvae_cfg, 
        object_pretrain,
        env_pretrain
    ):
        super(VQVAEEncoder, self).__init__()
        # object encoder
        self.object_vqvae = VQVAE(vqvae_cfg)
        object_ckpt = torch.load(object_pretrain, map_location='cpu', weights_only=False)
        self.object_vqvae.load_state_dict(object_ckpt)
        print(f"Loaded object pretrained VQVAE from {object_pretrain}.")
        for param in self.object_vqvae.parameters():
            param.requires_grad = False
        
        # env encoder
        self.env_vqvae = VQVAE(vqvae_cfg)
        env_ckpt = torch.load(env_pretrain, map_location='cpu', weights_only=False)
        self.env_vqvae.load_state_dict(env_ckpt)
        print(f"Loaded env pretrained VQVAE from {env_pretrain}.")
        for param in self.env_vqvae.parameters():
            param.requires_grad = False

    def _normalize_pc_(self, pc):

        # recenter
        B, N, _ = pc.shape
        centroids = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroids

        scale, _ = torch.max(torch.abs(pc), dim=1, keepdim=True)
        scale, _ = torch.max(scale, dim=2, keepdim=True)
        pc = pc / scale

        return pc, centroids, scale

    def forward(self, batch):

        object_pc = batch["object_pc"]
        env_pc = batch["env_pc"]

        # Env Node
        with torch.no_grad():
            normal_pc_env, env_centroids, env_scale = self._normalize_pc_(env_pc)
            env_tokens = self.env_vqvae.encode(normal_pc_env) 

        batch["env_xyz"] = env_tokens["xyz"]
        batch["env_feat"] = env_tokens["z_q"]
        batch["env_centroids"] = env_centroids
        batch["env_scale"] = env_scale
        
        # Object Node
        with torch.no_grad():
            normal_pc_object, object_centroids, object_scale = self._normalize_pc_(object_pc)
            object_tokens = self.object_vqvae.encode(normal_pc_object)

        object_xyz_local = object_tokens["xyz"]
        object_xyz_world = object_xyz_local * object_scale + object_centroids
        object_xyz_env = (object_xyz_world - env_centroids) / env_scale

        batch["object_xyz"] = object_xyz_env
        batch["object_feat"] = object_tokens["z_q"]
        
        return batch