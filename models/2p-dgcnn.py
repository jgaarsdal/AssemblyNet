import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, device=torch.device('cuda')):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class EdgeConv(nn.Module):
    def __init__(self, args, input_size, output_size, device=torch.device('cuda')):
        super(EdgeConv, self).__init__()
        self.args = args
        self.device = device
        self.k = args.n_knn if not args == None else 20
        self.drop_rate = 0.5
        
        self.conv1 = nn.Sequential(nn.Conv2d(input_size, output_size, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_size),
                                   nn.GELU())

    def forward(self, x):
        x = get_graph_feature(x, k=self.k, device=self.device)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        return x1
 
 
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


class TwoPathDGCNN(nn.Module):
    def __init__(self, args, device=torch.device('cuda')):
        super(TwoPathDGCNN, self).__init__()
        self.args = args
        self.device = device
        self.k = args.n_knn if not args == None else 20
        self.emb_dims = 1024
        self.drop_rate = 0.5

        self.p1_edgeconv1 = EdgeConv(args, 6, 64, device)
        self.p1_edgeconv2 = EdgeConv(args, 64*2, 64, device)
        self.p1_edgeconv3 = EdgeConv(args, 64*2, 128, device)
        self.p1_edgeconv4 = EdgeConv(args, 128*2, 256, device)
        
        self.p2_edgeconv1 = EdgeConv(args, 6, 64, device)
        self.p2_edgeconv2 = EdgeConv(args, 64*2, 64, device)
        self.p2_edgeconv3 = EdgeConv(args, 64*2, 128, device)
        self.p2_edgeconv4 = EdgeConv(args, 128*2, 256, device)

        self.crossattention1 = CrossAttention(64, 2)
        self.crossattention2 = CrossAttention(64, 2)
        self.crossattention3 = CrossAttention(128, 4)
        self.crossattention4 = CrossAttention(256, 8)

        self.decoder1 = nn.Sequential(nn.Linear(1024, 256, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.GELU())
        
        self.decoder2 = nn.Sequential(nn.Linear(768, 256, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.GELU())
        
        self.decoder3 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.GELU())
        
        self.decoder4 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.GELU())

    def forward(self, x):
        batch_size = x.size(0)

        p1_x1 = self.p1_edgeconv1(x[:,0,:,:])
        p2_x1 = self.p2_edgeconv1(x[:,1,:,:])
        cfeat1 = self.crossattention1(p1_x1, p2_x1, p2_x1)

        p1_x2 = self.p1_edgeconv2(p1_x1)
        p2_x2 = self.p2_edgeconv2(p2_x1)
        cfeat2 = self.crossattention2(p1_x2, p2_x2, p2_x2)

        p1_x3 = self.p1_edgeconv3(p1_x2)
        p2_x3 = self.p2_edgeconv3(p2_x2)
        cfeat3 = self.crossattention3(p1_x3, p2_x3, p2_x3)

        p1_x4 = self.p1_edgeconv4(p1_x3)
        p2_x4 = self.p2_edgeconv4(p2_x3)
        cfeat4 = self.crossattention4(p1_x4, p2_x4, p2_x4)

        fx = torch.cat((p1_x4, cfeat4), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fxM, fxA), 1)
        fx1 = self.decoder1(fx)

        fx = torch.cat((p1_x3, cfeat3), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fx1, fxM, fxA), 1)
        fx2 = self.decoder2(fx)

        fx = torch.cat((p1_x2, cfeat2), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fx2, fxM, fxA), 1)
        fx3 = self.decoder3(fx)

        fx = torch.cat((p1_x1, cfeat1), dim=1)
        fxM = F.adaptive_max_pool1d(fx, 1).view(batch_size, -1)
        fxA = F.adaptive_avg_pool1d(fx, 1).view(batch_size, -1)
        fx = torch.cat((fx3, fxM, fxA), 1)
        fx = self.decoder4(fx)

        return fx


class get_model(nn.Module):
    def __init__(self, args, output_channels=26, device=torch.device('cuda')):
        super(get_model, self).__init__()
        self.args = args
        self.device = device

        self.dual_dgcnn = TwoPathDGCNN(args, device)
        self.output = nn.Linear(256, output_channels)

    def forward(self, x):
        x = self.dual_dgcnn(x)
        x = self.output(x)
        return x

def get_no_params(model):
    nop = 0
    for param in list(model.parameters()):
        nn = 1
        for s in list(param.size()):
            nn = nn * s
        nop += nn
    return nop

if __name__ == '__main__':
    model = get_model(args=None)
    print('The number of parameters in the model are %d:' % get_no_params(model))