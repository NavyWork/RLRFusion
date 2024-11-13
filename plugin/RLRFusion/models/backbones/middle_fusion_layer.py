"""
Used for fuse the middle features of lidar and radar Hejiale
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from plugin.RLRFusion.core.gaussian import *
from mmdet3d.ops import DynamicScatter,ball_query,gather_points
from mmdet3d.ops import grouping_operation
from mmdet3d.models.utils import MLP
from mmcv.runner import force_fp32
# Initialize a Registry
from mmcv.utils import Registry
MIDDLE_FUSION_LAYERS =  Registry('middle_fusion_layer')

# Create a build function from config file
def build_middle_fusion_layer(cfg):
    """Build backbone."""
    return MIDDLE_FUSION_LAYERS.build(cfg)

@MIDDLE_FUSION_LAYERS.register_module()
class rcs_middle_fusion_gamma(nn.Module):
    def __init__(self, in_channels):
        super(rcs_middle_fusion_gamma, self).__init__()
        self.rcs_att = nn.Conv2d(2, in_channels, 1)
        self.compress = nn.Conv2d(in_channels*2, in_channels, 3, padding=1)
        self.rcs_gamma = nn.Parameter(torch.tensor(1.0))


    def forward(self, middle_feat_lidar, voxels_feat_radar,rcoors, batch_size, rcs):
        ### rcs_features
        heatmap = voxels_feat_radar.new_zeros((batch_size, 128, 128))
        heatmap_feat = voxels_feat_radar.new_zeros((batch_size, 1, 128, 128))

        true_rcs = torch.nn.functional.relu(rcs)
        radius = self.rcs_gamma * true_rcs + 1

        for i in range(rcoors.shape[0]):
            batch, _, y, x = rcoors[i]
            draw_heatmap_gaussian_rcs(heatmap[batch], [x, y], int(radius[i].data.item()))
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(heatmap_feat[batch], [x, y], int(radius[i].data.item()),
                                                             true_rcs[i])
        rcs_features = self.rcs_att(torch.cat([heatmap.unsqueeze(dim=1), heatmap_feat], dim=1))
        middle_feat_lidar = self.compress(torch.cat([middle_feat_lidar, rcs_features], dim=1))
        return middle_feat_lidar



