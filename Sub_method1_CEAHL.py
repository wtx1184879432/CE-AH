import torch.nn as nn
from collections import OrderedDict
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


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
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
        return local_feature


class CEAH(nn.Module):
    def __init__(self, patch_num=60):
        super(CEAH, self).__init__()
        self.patch_num = patch_num
        self.patch_net = BaseNet()
        self.reduce_channels = nn.Sequential(
            nn.Conv3d(60, 64, kernel_size=2),
            GroupNorm(num_features=64, num_groups=8),
            nn.ReLU(True),
            nn.Conv3d(64, 64, kernel_size=2),
            GroupNorm(num_features=64, num_groups=8),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        patch_feature, patch_score = [], []
        for i in range(self.patch_num):
            # [10, 128, 6, 6, 6]
            feature = self.patch_net(input[i])
            feature = feature.unsqueeze(1)
            patch_feature.append(feature)

        region_feature_maps = torch.cat(patch_feature, 1)  # 10 48 64 , 4 4 4
        region_feature_maps = region_feature_maps.mean(2)  # 10 48 4 4 4
        features = self.reduce_channels(region_feature_maps).flatten(1)
        subject_pred = self.fc_1(features)
        return subject_pred

