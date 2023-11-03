import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2 import index_points, query_ball_point, sample_and_group_all, sample_and_group, farthest_point_sample


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.gelu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat
    

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.gelu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

 
class CrossAttention(nn.Module):
    def __init__(self, num_features, num_heads):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.num_features = num_features
        assert num_features % self.num_heads == 0

        # Linear layers that transform the input into query, key, and value vectors
        self.query = nn.Linear(num_features, num_features)
        self.key = nn.Linear(num_features, num_features)
        self.value = nn.Linear(num_features, num_features)

        # Applied to the concatenated output of all attention heads
        self.output = nn.Linear(num_features, num_features)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        value = value.transpose(2, 1)

        # Transform the input (batch_size, seq_len, num_features)
        query = self.query(query).view(batch_size, -1, self.num_heads, self.num_features // self.num_heads).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.num_features // self.num_heads).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.num_features // self.num_heads).transpose(1, 2)

        # Compute dot product between query and key, divide by square root of the depth, apply the mask (if any) and softmax
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.num_features))
        attention = torch.softmax(scores, dim=-1)

        # Compute weighted sum of the value vectors
        out = torch.matmul(attention, value)

        # Rearrange and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * (self.num_features // self.num_heads))
        out = self.output(out)
        out = out.transpose(2, 1)
        return out
    

class TwoPathPointNet2(nn.Module):
    def __init__(self, args, device=torch.device('cuda')):
        super(TwoPathPointNet2, self).__init__()
        self.args = args
        self.device = device
        self.npoint = args.num_point
        self.drop_rate = 0.5

        self.p1_sa1 = PointNetSetAbstractionMsg(self.npoint, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.p1_sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.p1_sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.p2_sa1 = PointNetSetAbstractionMsg(self.npoint, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.p2_sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.p2_sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.crossattention1 = CrossAttention(320, 10)
        self.crossattention2 = CrossAttention(640, 20)
        self.crossattention3 = CrossAttention(1024, 32)

        self.decoder1 = nn.Sequential(nn.Linear(4096, 512, bias=False),
                                      nn.BatchNorm1d(512),
                                      nn.GELU())
        
        self.decoder2 = nn.Sequential(nn.Linear(3072, 512, bias=False),
                                      nn.BatchNorm1d(512),
                                      nn.GELU())
        
        self.decoder3 = nn.Sequential(nn.Linear(1792, 512, bias=False),
                                      nn.BatchNorm1d(512),
                                      nn.GELU())

    def forward(self, x):
        batch_size = x.size(0)

        p1_l1_xyz, p1_l1_points = self.p1_sa1(x[:,0,:,:], None)
        p2_l1_xyz, p2_l1_points = self.p2_sa1(x[:,1,:,:], None)
        cfeat1 = self.crossattention1(p1_l1_points, p2_l1_points, p2_l1_points)

        p1_l2_xyz, p1_l2_points = self.p1_sa2(p1_l1_xyz, p1_l1_points)
        p2_l2_xyz, p2_l2_points = self.p2_sa2(p2_l1_xyz, p2_l1_points)
        cfeat2 = self.crossattention2(p1_l2_points, p2_l2_points, p2_l2_points)

        _, p1_l3_points = self.p1_sa3(p1_l2_xyz, p1_l2_points)
        _, p2_l3_points = self.p2_sa3(p2_l2_xyz, p2_l2_points)
        cfeat3 = self.crossattention3(p1_l3_points, p2_l3_points, p2_l3_points)

        fx = torch.cat((p1_l3_points, cfeat3), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fxM, fxA), 1)
        fx1 = self.decoder1(fx)

        fx = torch.cat((p1_l2_points, cfeat2), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fx1, fxM, fxA), 1)
        fx2 = self.decoder2(fx)

        fx = torch.cat((p1_l1_points, cfeat1), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fx2, fxM, fxA), 1)
        fx3 = self.decoder3(fx)

        return fx3


class get_model(nn.Module):
    def __init__(self, args, output_channels=26, device=torch.device('cuda')):
        super(get_model, self).__init__()
        self.args = args
        self.device = device

        self.dual_pointnet2 = TwoPathPointNet2(args, device)
        self.output = nn.Linear(512, output_channels)

    def forward(self, x):
        x = self.dual_pointnet2(x)
        x = self.output(x)
        return x