import numpy as np
import torch
import torch.nn as nn
from typing import OrderedDict

from model.common.base_nets import ResNet18Conv, SpatialSoftmax
from model.diffusion.cond_unet import replace_bn_with_gn, ConditionalUnet1D

    
class ResNetBackbone(nn.Module):
    def __init__(
        self, feature_dim, 
        pretrained=True, num_images=1, num_kp=32
    ):
        super().__init__()
        backbones = []
        pools = []
        linears = []
        for _ in range(num_images):
            backbones.append(
                ResNet18Conv(
                    input_channel=3,
                    pretrained=pretrained,
                    input_coord_conv=False
                )
            )
            pools.append(
                SpatialSoftmax(
                    input_shape=[512, 7, 7],  # for (224, 224)
                    num_kp=num_kp,
                    temperature=1.0,
                    learnable_temperature=False,
                    noise_std=0.0
                )
            )
            linears.append(
                torch.nn.Linear(int(np.prod([num_kp, 2])), feature_dim)
            )

        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        backbones = replace_bn_with_gn(backbones)  # TODO
        self.backbones = backbones
        self.pools = pools
        self.linears = linears



class ResNetImageStateEmbeddung(nn.Module):
    def __init__(
        self, 
        resnet_feature_dim, 
        resnet_pretrained=True, 
        num_images=1, num_kp=32,
    ):
        super().__init__()
        self.image_encoder = ResNetBackbone(
            feature_dim=resnet_feature_dim, 
            pretrained=resnet_pretrained, 
            num_images=num_images, num_kp=num_kp
        )
        
        
    def forward(self, qpos, image):
        all_features = []
        T_o = qpos.shape[1]
        for t_idx in range(T_o):
            time_features = []
            cam_image = image[:, t_idx] 
            # only support 1 image for now, so cam_image is (B, C, H, W)
            cam_features = self.image_encoder.backbones[0](cam_image)
            pool_features = self.image_encoder.pools[0](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = self.image_encoder.linears[0](pool_features)
            time_features.append(out_features)
            
            time_obs = torch.cat(time_features + [qpos[:, t_idx]], dim=1)
            all_features.append(time_obs)
        obs_cond = torch.cat(all_features, dim=1)
        return obs_cond
    
    def load_ckpt(self, ckpt_path):
        model_dict = torch.load(ckpt_path, weights_only=True)
        
        noise_pred_net_dict = OrderedDict()
        backbone_dict = OrderedDict()
        pools_dict = OrderedDict()
        linears_dict = OrderedDict()
        for key, value in model_dict["nets"].items():
            if key.startswith("policy.noise_pred_net"):
                noise_pred_net_dict.update({key[len("policy.noise_pred_net."):]: value})
            if key.startswith("policy.backbones"):
                backbone_dict.update({key[len("policy.backbones."):]: value})
            if key.startswith("policy.pools"):
                pools_dict.update({key[len("policy.pools."):]: value})
            if key.startswith("policy.linears"):
                linears_dict.update({key[len("policy.linears."):]: value})
        
        self.image_encoder.backbones.load_state_dict(backbone_dict)
        self.image_encoder.pools.load_state_dict(pools_dict)
        self.image_encoder.linears.load_state_dict(linears_dict)
        
        