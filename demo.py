# from PointNet2 import kan
# import numpy as np
# import torch
# import torch.nn as nn
#
# # 生成随机的2维矩阵
# input_matrix = np.random.rand(5, 9)  # 修改输入矩阵的形状为 (5, 9)
#
# # 定义简单的多层感知器（MLP）模型
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
#        # self.fc1=kan.KANLinear(input_size, hidden_size)
#         #self.fc2 = nn.Linear(hidden_size, input_size)  # 隐藏层到输出层
#         #self.fc2 = kan.KANLinear(input_size, hidden_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  # 使用ReLU激活函数进行隐藏层变换
#       #  x = self.fc2(x)  # 进行输出层变换
#         return x
#
# # 定义输入大小和隐藏层大小
# input_size = 9  # 修改输入矩阵的大小为 9
# hidden_size = 64  # 隐藏层大小
#
# # 初始化模型
# model = SimpleMLP(input_size, hidden_size)
#
# # 转换输入矩阵为 PyTorch Tensor
# input_tensor = torch.tensor(input_matrix, dtype=torch.float32)
#
# # 运行模型
# output_matrix = model(input_tensor).detach().numpy()
#
# print("输入矩阵：")
# print(input_matrix)
# print("\n经过MLP变换后的输出矩阵：")
# print(output_matrix)








import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from block import pointnet2_utils

"""
标准空间卷积假设输入数据具有规则的邻域结构。现有方法通常通过例如固定邻域大小来固定规则“视图”，从而将卷积推广到不规则点云域，其中每个点的卷积核大小保持相同。
然而，由于点云不像图像那样结构化，固定的邻居数会产生不幸的归纳偏差。我们提出了一种新颖的图卷积，称为差异图卷积（diffConv），它不依赖于常规视图。 
diffConv 在空间变化和密度扩张的邻域上运行，这些邻域通过学习的屏蔽注意力机制进一步适应。
实验表明，我们的模型对噪声非常鲁棒，在 3D 形状分类和场景理解任务中获得了最先进的性能，并且推理速度更快。
"""


def get_dist(src, dst):
    """
    Calculate the Euclidean distance between each point pair in two point clouds.
    Inputs:
        src[B, M, 3]: point cloud 1
        dst[B, N, 3]: point cloud 2
    Return:
        dist[B, M, N]: distance matrix
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points[B, N, C]: input point features
        idx[B, M]: sample index data
    Return:
        new_points[B, M, C]: quried point features
    """
    new_points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    return new_points


def sample_and_group(radius, k, xyz, feat, centroid, dist):
    """
    Input:
        radius[B, M, 1]: search radius of each key point
        k(int): max number of samples in local region
        xyz[B, N, 3]: query points
        centroid[B, M, 3]: key points
        dist[B, M, N]: distance matrix
        feat[B, N, D]: input points features
    Return:
        cent_feat[B, M, D]: grouped features
        idx[B, M, k]: indices of selected neighbors
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, M, _ = centroid.shape

    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])

    idx[dist > radius] = N
    idx = idx.sort(dim=-1)[0][:, :, :k]
    group_first = idx[:, :, 0].view(B, M, 1).repeat([1, 1, k])
    mask = (idx == N)
    idx[mask] = group_first[mask]

    torch.cuda.empty_cache()
    idx = idx.int().contiguous()

    feat = feat.transpose(1, 2).contiguous()
    cent_feat = pointnet2_utils.grouping_operation(feat, idx)
    cent_feat = cent_feat.transpose(1, 2).transpose(-1, -2).contiguous()
    torch.cuda.empty_cache()

    return cent_feat, idx

class Conv1x1(nn.Module):
    '''
    1x1 1d convolution
    '''

    def __init__(self, in_channels, out_channels, act=nn.GELU(), bias_=False):  # nn.LeakyReLU(negative_slope=0.2)
        super(Conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias_),
            nn.BatchNorm1d(out_channels)
        )
        self.act = act
        nn.init.xavier_normal_(self.conv[0].weight.data)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)

        x = x.transpose(1, 2).contiguous()
        if self.act is not None:
            return self.act(x)
        else:
            return x


class PositionEncoder(nn.Module):
    def __init__(self, out_channel, radius, k=20):
        super(PositionEncoder, self).__init__()
        self.k = k

        self.xyz2feature = nn.Sequential(
            nn.Conv2d(9, out_channel // 8, kernel_size=1),
            nn.BatchNorm2d(out_channel // 8),
            nn.GELU()
        )

        self.mlp = nn.Sequential(
            Conv1x1(out_channel // 8, out_channel // 4),
            Conv1x1(out_channel // 4, out_channel, act=None)
        )

        self.qg = pointnet2_utils.QueryAndGroup(radius, self.k)

    def forward(self, centroid, xyz, radius, dist):
        point_feature, _ = sample_and_group(radius, self.k, xyz, xyz, centroid, dist)  # [B, N, k, 3]

        points = centroid.unsqueeze(2).repeat(1, 1, self.k, 1)  # [B, N, k, 3]

        variance = point_feature - points  # [B, N, k, 3]

        point_feature = torch.cat((points, point_feature, variance), dim=-1)  # [B, N, k, 9]

        point_feature = point_feature.permute(0, 3, 1, 2).contiguous()  # [B, 9, N, k]

        point_feature = self.xyz2feature(point_feature)  # [B, 9, N, k]

        point_feature = torch.max(point_feature, dim=-1)[0].transpose(1, 2)  # [B, N, C]

        point_feature = self.mlp(point_feature)  # [B, N, C']

        return point_feature


class MaskedAttention(nn.Module):
    def __init__(self, in_channels, hid_channels=128):
        super().__init__()
        if not hid_channels:
            hid_channels = 1
        self.conv_q = Conv1x1(in_channels + 3, hid_channels,
                              act=None)  # map query (key points) to another linear space
        self.conv_k = Conv1x1(in_channels + 3, hid_channels,
                              act=None)  # map key (neighbor points) to another linear space

    def forward(self, cent_feat, feat, mask):
        '''
        Inputs:
            cent_feat: [B, M, C+3]
            feat: [B, N, C+3]
            mask: [B, M, N]

        Returns:
            adj: [B, M, N]
        '''
        q = self.conv_q(cent_feat)  # [B, M, C+3] -> [B, M, C_int]

        k = self.conv_k(feat)  # [B, N, C+3] -> [B, N, C_int]

        adj = torch.bmm(q, k.transpose(1, 2))  # [B, M, C_int] * [B, C_int, N] -> [B, M, N]

        # masked self-attention: masking all non-neighbors (Eq. 9)
        adj = adj.masked_fill(mask < 1e-9, -1e9)
        adj = torch.softmax(adj, dim=-1)

        # balanced renormalization (Eq. 11)
        adj = torch.sqrt(mask + 1e-9) * torch.sqrt(adj + 1e-9) - 1e-9

        adj = F.normalize(adj, p=1, dim=1)  # [B, M, N]
        adj = F.normalize(adj, p=1, dim=-1)  # [B, M, N]

        return adj


def dilated_ball_query(dist, h, base_radius, max_radius):
    '''
    Density-dilated ball query
    Inputs:
        dist[B, M, N]: distance matrix
        h(float): bandwidth
        base_radius(float): minimum search radius
        max_radius(float): maximum search radius
    Returns:
        radius[B, M, 1]: search radius of point
    '''

    # kernel density estimation (Eq. 8)
    sigma = 1
    gauss = torch.exp(-(dist) / (2 * (h ** 2) * (sigma ** 2)))  # K(x-x_i/h), [B, M, N]
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1)  # kernel distance, [B, M, 1]

    # normalization
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9)  # [B, M, 1]
    radius = base_radius + (max_radius - base_radius) * kd_score  # kd_score -> max, base_radius -> max_radius

    return radius


class diffConv(nn.Module):
    def __init__(self, in_channels, out_channels, base_radius, bottleneck=4):
        super().__init__()
        self.conv_v = Conv1x1(2 * in_channels, out_channels, act=None)
        self.mat = MaskedAttention(in_channels, in_channels // bottleneck)
        self.pos_conv = PositionEncoder(out_channels, np.sqrt(base_radius))
        self.base_radius = base_radius  # squared radius

    def forward(self, x, xyz, cent_num):
        '''
        Inputs:
            x[B, N, C]: point features
            xyz[B, N, 3]: points
            cent_num(int): number of key points

        Returns:
            x[B, M, C']: updated point features
            centroid[B, M, 3]: sampled features
        '''
        batch_size, point_num = xyz.size(0), xyz.size(1)

        if cent_num < point_num:
            # random sampling
            idx = np.arange(point_num)
            idx = idx[:cent_num]
            idx = torch.from_numpy(idx).unsqueeze(0).repeat(batch_size, 1).int().to(xyz.device)

            # gathering
            centroid = index_points(xyz, idx)  # [B, M, 3]
            cent_feat = index_points(x, idx)  # [B, M, C]
        else:
            centroid = xyz.clone()
            cent_feat = x.clone()

        dist = get_dist(centroid, xyz)  # disntance matrix, [B, M, N]

        radius = dilated_ball_query(dist, h=0.1, base_radius=self.base_radius, max_radius=self.base_radius * 2)

        mask = (dist < radius).float()

        # get attentive mask (adjacency matrix)
        emb_cent = torch.cat((cent_feat, centroid), dim=-1)
        emb_x = torch.cat((x, xyz), dim=-1)
        adj = self.mat(emb_cent, emb_x, mask)  # [B, M, N]

        # inner-group attention
        smoothed_x = torch.bmm(adj, x)  # [B, M, N] * [B, N, C] -> [B, M, C]
        variation = smoothed_x - cent_feat  # [B, M, C] -> [B, M, C]

        x = torch.cat((variation, cent_feat), dim=-1)  # [B, M, C] -> [B, M, 2C]
        x = self.conv_v(x)  # [B, M, 2C] -> [B, M, C']

        pos_emb = self.pos_conv(centroid, xyz, radius, dist)

        # feature fusion
        x = x + pos_emb
        x = F.gelu(x)

        return x, centroid


class Attention_block(nn.Module):
    '''
    attention U-Net is taken from https://github.com/tiangexiang/CurveNet/blob/main/core/models/curvenet_util.py.
    '''

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g = g.transpose(1, 2)
        x = x.transpose(1, 2)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.gelu(g1 + x1)
        psi = self.psi(psi)
        psi = psi.transpose(1, 2)

        return psi


class PointFeaturePropagation(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(PointFeaturePropagation, self).__init__()
        in_channel = in_channel1 + in_channel2
        self.conv = nn.Sequential(
            Conv1x1(in_channel, in_channel // 2),
            Conv1x1(in_channel // 2, in_channel // 2),
            Conv1x1(in_channel // 2, out_channel)
        )
        self.att = Attention_block(in_channel1, in_channel2, in_channel2)

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, M, 3]
            feat1: input points data, [B, N, C']
            feat2: input points data, [B, M, C]
        Return:
            new_points: upsampled points data, [B, N, C+C']
        """
        dists, idx = pointnet2_utils.three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dists + 1e-8)  # [B, N, 3]
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)  # [B, N, 1]
        weight = dist_recip / norm  # [B, N, 1]
        int_feat = pointnet2_utils.three_interpolate(feat2.transpose(1, 2).contiguous(), idx, weight).transpose(1, 2)

        psix = self.att(int_feat, feat1)
        feat1 = feat1 * psix

        if feat1 is not None:
            cat_feat = torch.cat((feat1, int_feat), dim=-1)  # [B, N, C'], [B, N, C] -> [B, N, C + C']
        else:
            cat_feat = int_feat  # [B, N, C]
        cat_feat = self.conv(cat_feat)  # [B, N, C + C'] -> [B, N, C']

        return cat_feat


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = torch.rand(5, 1024, 3).to(device)
    feat = torch.rand(5, 1024, 16).to(device)
    model = diffConv(16, 32, 0.1).to(device)
    new_feat, new_pc = model(feat, pc, 512)
    print(new_feat.shape, new_pc.shape)

























