import torch.nn as nn
import einops

from model.common.critic import CriticObs
from model.resnet.base_nets import ResNetImageStateEmbeddung
from model.common.modules import RandomShiftsAug


class ResNetCritic(CriticObs):
    def __init__(
        self,
        backbone_ckpt=None,
        augment=False, 
        img_cond_steps=1,
        feature_dim=64,
        mlp_obs_dim=74, # TODO: convenient compute
        **kwargs
    ):
        
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.obs_encoder = ResNetImageStateEmbeddung(
            resnet_feature_dim=feature_dim, num_images=img_cond_steps
        )
        self.num_img=img_cond_steps
        if backbone_ckpt is not None:
            self.obs_encoder.load_ckpt(backbone_ckpt)
            
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        
        
    def aug_rgb(self, rgb):
        B, T_rgb, C, H, W = rgb.shape
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        
        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            rgb1 = self.aug(rgb1)
            rgb2 = self.aug(rgb2)
        else:  # single image
            rgb = self.aug(rgb)  # uint8 -> float32
        
        if self.num_img > 1:
            rgb1 = rgb1.reshape(B, T_rgb, 3, H, W)
            rgb2 = rgb2.reshape(B, T_rgb, 3, H, W)
            return rgb1, rgb2
        else:
            rgb = rgb.reshape(B, T_rgb, C, H, W)
            return rgb
        
    def forward(self, cond: dict, no_augment=False):
        if cond is None:
            raise ValueError("condition must be provided")
            
        qpos = cond.get('state')  # (B, To, Do)
        image = cond.get('rgb')   # (B, To, C, H, W)
        
        if qpos is None or image is None:
            raise ValueError("condition must include 'state' and 'rgb' keys")
        
        if self.augment and not no_augment:
            image = self.aug_rgb(image)
            
        obs_cond = self.obs_encoder(qpos, image)  # [1, 74]
        return super().forward(obs_cond)
        