import torch.nn as nn
import torch_geometric.nn.conv.transformer_conv as trans_conv
import torch_geometric.nn.norm as geo_norm


class Inter_fuse_module(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.geo_0 = trans_conv.TransformerConv(in_dim, out_dim)
        self.norm_0 = geo_norm.LayerNorm(out_dim)
        self.relu_0 = nn.ReLU(True)

    def forward(self, geo_x, euc_x, edge_index):

        x = (geo_x, euc_x)
        out = self.geo_0(x, edge_index)
        out = self.relu_0(self.norm_0(out))
        
        return out


class Intra_aggr_module(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.geo_0 = trans_conv.TransformerConv(in_dim, out_dim)
        self.norm_0 = geo_norm.LayerNorm(out_dim)
        self.relu_0 = nn.ReLU(True)

        self.geo_1 = trans_conv.TransformerConv(out_dim, out_dim)
        self.norm_1 = geo_norm.LayerNorm(out_dim)
        self.relu_1 = nn.ReLU(True)

    def forward(self, data):
        # input
        x, edge_index = data.x, data.edge_index

        x_0 = self.geo_0(x, edge_index)
        x_0 = self.relu_0(self.norm_0(x_0)) + x

        x_1 = self.geo_1(x_0, edge_index)
        x_1 = self.relu_1(self.norm_1(x_1)) + x_0

        return x_1
    