import torch
from torch import nn
from torch.nn import functional as F

from nn_zoo.models.components import DepthwiseSeparableConv2d


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(1, 4, 3, 2, 1)
        self.conv2 = DepthwiseSeparableConv2d(4, 8, 3, 2, 1)
        self.conv3 = DepthwiseSeparableConv2d(8, 16, 3, 2, 1)
        self.conv4 = DepthwiseSeparableConv2d(16, 32, 3, 2, 1)
        self.conv5 = DepthwiseSeparableConv2d(32, 64, 3, 2, 1)

        self.fc1 = nn.Linear(64, 10)

        self.load_state_dict(torch.load("model.pth"))

    def forward(self, x, lpips=False):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        x5 = x5.view(x5.size(0), -1)

        x = self.fc1(x5)

        if lpips:
            return x, x1, x2, x3, x4

        return x

    def _lpip(self, feature1, feature2):
        # Normalize feature maps
        f1_mu = feature1.mean(dim=[2, 3], keepdim=True)
        f1_std = feature1.std(dim=[2, 3], keepdim=True)
        norm_feature1 = (feature1 - f1_mu) / (f1_std + 1e-8)

        f2_mu = feature2.mean(dim=[2, 3], keepdim=True)
        f2_std = feature2.std(dim=[2, 3], keepdim=True)
        norm_feature2 = (feature2 - f2_mu) / (f2_std + 1e-8)

        # Calculate the squared difference
        diff = (norm_feature1 - norm_feature2) ** 2

        # Spatial average
        avg_diff = diff.flatten(start_dim=1).mean(dim=1)

        # Average across channels
        avg = avg_diff.mean()

        return avg

    def lpips(self, img1, img2, linear=False):
        # Forward pass with lpips=True to get feature maps
        features1 = self.forward(img1, lpips=True)
        features2 = self.forward(img2, lpips=True)

        # Calculate the LPIPS score
        if linear:
            scores = list(map(lambda x, y: self._lpip(x, y), features1, features2))
        else:
            scores = list(
                map(lambda x, y: self._lpip(x, y), features1[1:], features2[1:])
            )

        # Sum the distances to get the LPIPS score
        lpips_score = sum(scores)

        return lpips_score.mean().flatten()
