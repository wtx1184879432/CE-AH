import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cs = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        # print(a.shape) (torch.Size([10, 2, 6, 6, 6]))
        a = self.cs(a)
        # print(a.shape) (torch.Size([10, 1, 6, 6, 6]))
        return x * a

class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.GlobalAveragePool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.GlobalMaxPool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        self.Attn = nn.Sequential(
            nn.Conv3d(48, 48 // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(48 // 2, 48, kernel_size=1)
        )

    def forward(self, input, patch_pred):
        mean_input = input
        # 10 60 , 1, 2, 2
        attn1 = self.Attn(self.GlobalAveragePool(mean_input))
        attn2 = self.Attn(self.GlobalMaxPool(mean_input))

        patch_pred = patch_pred.unsqueeze(-1)

        patch_pred = patch_pred.unsqueeze(-1)

        patch_pred = patch_pred.unsqueeze(-1)

        a = attn1 + attn2 + patch_pred
        a = torch.sigmoid(a)
        return mean_input * a

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.features = nn.Sequential(OrderedDict([

            ('conv1', nn.Conv3d(1, 16, kernel_size=4)),
            ('BatchNorm3d1', nn.BatchNorm3d(16)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(16, 32, kernel_size=3)),
            ('BatchNorm3d2', nn.BatchNorm3d(32)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=2)),
            ('conv3', nn.Conv3d(32, 48, kernel_size=3)),
            ('BatchNorm3d3', nn.BatchNorm3d(48)),
            ('relu3', nn.ReLU(inplace=True)),

            ('drop1', nn.Dropout(0.5)),
        ]))

    def forward(self, input):
        local_feature = self.features(input)
        attended_feature = self.spatial_attention(local_feature)
        return attended_feature


class RegionNet(nn.Module):
    def __init__(self):
        super(RegionNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            # 10 * 2 *, 1, 2, 2
            ('conv8', nn.Conv3d(48, 48, kernel_size=3)),
            ('BatchNorm3d8', nn.BatchNorm3d(48)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv3d(48, 64, kernel_size=3)),
            ('BatchNorm3d9', nn.BatchNorm3d(64)),
            ('relu9', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.5)),
        ]))

        self.classify = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        local_feature = self.features(input)  # 64 9 4 4
        feature_ = F.adaptive_avg_pool3d(local_feature, (1, 1, 1))

        score = self.classify(feature_.flatten(1, -1))

        return [local_feature, score]


class CEAH(nn.Module):
    def __init__(self, patch_num=60):
        super(CEAH, self).__init__()
        self.patch_num = patch_num
        self.patch_net = BaseNet()
        self.region_net = RegionNet()
        self.attention_net = AttentionBlock()
        self.add_module('flat', Flatten())
        self.res_1 = nn.Sequential(
            nn.Conv3d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(True),
        )
        self.feature1 = nn.Sequential(
            nn.Linear(128 * 3 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
        )
        self.xyz = nn.Sequential(
            nn.Conv3d(48, 128, kernel_size=2),
            nn.BatchNorm3d(128),
        )

        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):

        reg_mar = [(0, 26), (1, 6), (2, 11), (3, 7), (4, 33), (5, 29), (6, 9), (7, 39), (8, 32), (9, 30), (10, 18), (11, 26), (12, 16), (13, 18), (14, 55), (15, 32), (16, 27), (17, 24), (18, 26), (19, 47), (20, 31), (21, 26), (22, 40), (23, 48), (24, 28), (25, 40), (26, 43), (27, 55), (28, 31), (29, 59), (30, 51), (31, 40), (32, 47), (33, 44), (34, 42), (35, 44), (36, 54), (37, 52), (38, 44), (39, 59), (44, 46), (45, 54), (48, 58), (49, 50), (51, 55), (52, 53), (53, 57), (57, 59)]


        patch_feature, region_feature, patch_score = [], [], []
        for i in range(self.patch_num):
            # [10, 128, 6, 6, 6]
            feature = self.patch_net(input[i])

            patch_feature.append(feature)

        for i in range(48):
            # b * 128 * 6 * 6 6
            reg = patch_feature[reg_mar[i][0]] + patch_feature[reg_mar[i][1]]

            # b * 128 * 4 4 4
            region_feature_unit, score = self.region_net(reg)
            # 10 1 64 , 4 4 4
            region_feature_unit = region_feature_unit.unsqueeze(1)

            region_feature.append(region_feature_unit)
            patch_score.append(score)

        region_feature_maps = torch.cat(region_feature, 1)  # 10 48 64 , 4 4 4
        patch_scores = torch.cat(patch_score, 1)
        region_feature_maps = region_feature_maps.mean(2)  # 10 48 4 4 4
        RRES = self.res_1(region_feature_maps)
        attn_feat = self.attention_net(RRES, patch_scores)  # 10 48  4 4 4
        attn_feat = region_feature_maps + attn_feat
        attn_feat = self.relu(attn_feat)
        attn_feat_1 = self.xyz(attn_feat)
        flat = self.flat(attn_feat_1)
        feature1 = self.feature1(flat)
        return feature1, flat

