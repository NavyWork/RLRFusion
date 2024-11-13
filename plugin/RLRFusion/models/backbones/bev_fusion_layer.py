"""
Used for fuse the middle features of lidar and radar weightedly
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from plugin.RLRFusion.core.gaussian import *

# Initialize a Registry
from mmcv.utils import Registry
BEV_FUSION_LAYERS =  Registry('bev_fusion_layer')

# Create a build function from config file
def build_bev_fusion_layer(cfg):
    """Build backbone."""
    return BEV_FUSION_LAYERS.build(cfg)


@BEV_FUSION_LAYERS.register_module()
class DQMITBF_MultiScale(nn.Module):
    def __init__(self, in_channels_1, H_1, W_1,
                 in_channels_2, H_2, W_2):
        super(DQMITBF_MultiScale, self).__init__()

        # Scale 1: (B, 128, 128, 128)
        self.learnable_query_1 = nn.Parameter(torch.randn(1, in_channels_1, H_1, W_1))
        self.conv3x3_1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=3, padding=1)

        # Scale 2: (B, 256, 64, 64)
        self.learnable_query_2 = nn.Parameter(torch.randn(1, in_channels_2, H_2, W_2))
        self.conv3x3_2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=3, padding=1)

    def forward(self, radar_bev, lidar_bev):
        """
        radar_bev: list of radar BEV feature maps at two scales [(batch, 128, 128, 128), (batch, 256, 64, 64)]
        lidar_bev: list of LiDAR BEV feature maps at two scales [(batch, 128, 128, 128), (batch, 256, 64, 64)]
        """
        # Scale 1 processing
        concatenated_features_1 = torch.cat([radar_bev[0], lidar_bev[0]],
                                            dim=1)  # Concatenate radar and lidar at scale 1
        concatenated_flat_1 = concatenated_features_1.flatten(2)
        learnable_query_flat_1 = self.learnable_query_1.flatten(2).expand(concatenated_flat_1.size(0), -1, -1)

        # Attention mechanism for Scale 1
        attention_weights_1 = F.softmax(torch.matmul(learnable_query_flat_1.transpose(1, 2), concatenated_flat_1),
                                        dim=-1)
        attention_output_1 = torch.matmul(attention_weights_1, concatenated_flat_1.transpose(1, 2))
        attention_output_1 = attention_output_1.view_as(concatenated_features_1)

        # Apply 3x3 convolution on the attention output for scale 1
        fused_output_1 = self.conv3x3_1(attention_output_1 * concatenated_features_1 + concatenated_features_1)

        # Scale 2 processing
        concatenated_features_2 = torch.cat([radar_bev[1], lidar_bev[1]],
                                            dim=1)  # Concatenate radar and lidar at scale 2
        concatenated_flat_2 = concatenated_features_2.flatten(2)
        learnable_query_flat_2 = self.learnable_query_2.flatten(2).expand(concatenated_flat_2.size(0), -1, -1)

        # Attention mechanism for Scale 2
        attention_weights_2 = F.softmax(torch.matmul(learnable_query_flat_2.transpose(1, 2), concatenated_flat_2),
                                        dim=-1)
        attention_output_2 = torch.matmul(attention_weights_2, concatenated_flat_2.transpose(1, 2))
        attention_output_2 = attention_output_2.view_as(concatenated_features_2)

        # Apply 3x3 convolution on the attention output for scale 2
        fused_output_2 = self.conv3x3_2(attention_output_2 * concatenated_features_2 + concatenated_features_2)

        # Return the fused outputs for both scales
        fused_output = [fused_output_1, fused_output_2]
        return fused_output