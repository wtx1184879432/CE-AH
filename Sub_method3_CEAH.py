import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, D, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


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


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.features = nn.Sequential(OrderedDict([

            ('conv1', nn.Conv3d(1, 16, kernel_size=4)),
            ('GroupNorm1', GroupNorm(num_features=16, num_groups=4)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(16, 32, kernel_size=3)),
            ('GroupNorm2', GroupNorm(num_features=32, num_groups=4)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=2)),
            ('conv3', nn.Conv3d(32, 48, kernel_size=3)),
            ('GroupNorm3', GroupNorm(num_features=48, num_groups=6)),
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
            ('GroupNorm8', GroupNorm(num_features=48, num_groups=6)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv3d(48, 64, kernel_size=3)),
            ('GroupNorm9', GroupNorm(num_features=64, num_groups=8)),
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
        self.res_1 = nn.Sequential(
            nn.Conv3d(48, 48, kernel_size=3, padding=1),
            GroupNorm(num_features=48, num_groups=6),
            nn.ReLU(True),
        )
        self.reduce_channels = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=2),
            GroupNorm(num_features=64, num_groups=8),
            nn.ReLU(True),
            nn.Conv3d(64, 64, kernel_size=2),
            GroupNorm(num_features=64, num_groups=8),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1),
        )

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
        region_feature_maps = region_feature_maps.mean(2)  # 10 48 4 4 4
        features = self.reduce_channels(region_feature_maps).flatten(1)
        subject_pred = self.fc_1(features)
        return subject_pred

