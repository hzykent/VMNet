import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor, PointTensor
import torch_geometric.nn.norm as geo_norm

from network.modules import Intra_aggr_module, Inter_fuse_module
from network.utils import voxel_to_point


class Euc_branch(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # Unet-like structure
        #-------------------------------- Input ----------------------------------------
        self.input_conv = nn.Sequential(
            spnn.Conv3d(3, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        #-------------------------------- Encoder ----------------------------------------
        # Level 0
        self.en0 = nn.Sequential(
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True),
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        self.down_0 = nn.Sequential(
            spnn.Conv3d(32, 64, 2, 2),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        # Level 1
        self.en1 = nn.Sequential(
            spnn.Conv3d(64, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True),
            spnn.Conv3d(64, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        self.down_1 = nn.Sequential(
            spnn.Conv3d(64, 96, 2, 2),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        # Level 2
        self.en2 = nn.Sequential(
            spnn.Conv3d(96, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True),
            spnn.Conv3d(96, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        self.down_2 = nn.Sequential(
            spnn.Conv3d(96, 128, 2, 2),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        # Level 3
        self.en3 = nn.Sequential(
            spnn.Conv3d(128, 128, 3, 1),
            spnn.BatchNorm(128),
            spnn.ReLU(True),
            spnn.Conv3d(128, 128, 3, 1),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        self.down_3 = nn.Sequential(
            spnn.Conv3d(128, 160, 2, 2),
            spnn.BatchNorm(160),
            spnn.ReLU(True)
        )

        # Level 4
        self.en4 = nn.Sequential(
            spnn.Conv3d(160, 160, 3, 1),
            spnn.BatchNorm(160),
            spnn.ReLU(True),
            spnn.Conv3d(160, 160, 3, 1),
            spnn.BatchNorm(160),
            spnn.ReLU(True)
        )

        self.down_4 = nn.Sequential(
            spnn.Conv3d(160, 192, 2, 2),
            spnn.BatchNorm(192),
            spnn.ReLU(True)
        )

        # Level 5
        self.en5 = nn.Sequential(
            spnn.Conv3d(192, 192, 3, 1),
            spnn.BatchNorm(192),
            spnn.ReLU(True),
            spnn.Conv3d(192, 192, 3, 1),
            spnn.BatchNorm(192),
            spnn.ReLU(True)
        )

        self.down_5 = nn.Sequential(
            spnn.Conv3d(192, 224, 2, 2),
            spnn.BatchNorm(224),
            spnn.ReLU(True)
        )

        #-------------------------------- Middle ----------------------------------------
        self.mid = nn.Sequential(
            spnn.Conv3d(224, 224, 3, 1),
            spnn.BatchNorm(224),
            spnn.ReLU(True),
            spnn.Conv3d(224, 224, 3, 1),
            spnn.BatchNorm(224),
            spnn.ReLU(True)
        )

        #-------------------------------- Decoder ----------------------------------------
        # Level 5
        self.up_5 = nn.Sequential(
            spnn.Conv3d(224, 192, 2, 2, transpose=True),
            spnn.BatchNorm(192),
            spnn.ReLU(True)
        )

        self.lin_net_5 = spnn.Conv3d(384, 192, kernel_size=1, stride=1, bias=False)
        self.de5 = nn.Sequential(
            spnn.Conv3d(384, 192, 3, 1),
            spnn.BatchNorm(192),
            spnn.ReLU(True),
            spnn.Conv3d(192, 192, 3, 1),
            spnn.BatchNorm(192),
            spnn.ReLU(True)
        )

        # Level 4
        self.up_4 = nn.Sequential(
            spnn.Conv3d(192, 160, 2, 2, transpose=True),
            spnn.BatchNorm(160),
            spnn.ReLU(True)
        )

        self.lin_net_4 = spnn.Conv3d(320, 160, kernel_size=1, stride=1, bias=False)
        self.de4 = nn.Sequential(
            spnn.Conv3d(320, 160, 3, 1),
            spnn.BatchNorm(160),
            spnn.ReLU(True),
            spnn.Conv3d(160, 160, 3, 1),
            spnn.BatchNorm(160),
            spnn.ReLU(True)
        )

        # Level 3
        self.up_3 = nn.Sequential(
            spnn.Conv3d(160, 128, 2, 2, transpose=True),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        self.lin_net_3 = spnn.Conv3d(256, 128, kernel_size=1, stride=1, bias=False)
        self.de3 = nn.Sequential(
            spnn.Conv3d(256, 128, 3, 1),
            spnn.BatchNorm(128),
            spnn.ReLU(True),
            spnn.Conv3d(128, 128, 3, 1),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        # Level 2
        self.up_2 = nn.Sequential(
            spnn.Conv3d(128, 96, 2, 2, transpose=True),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        self.lin_net_2 = spnn.Conv3d(192, 96, kernel_size=1, stride=1, bias=False)
        self.de2 = nn.Sequential(
            spnn.Conv3d(192, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True),
            spnn.Conv3d(96, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        # Level 1
        self.up_1 = nn.Sequential(
            spnn.Conv3d(96, 64, 2, 2, transpose=True),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        self.lin_net_1 = spnn.Conv3d(128, 64, kernel_size=1, stride=1, bias=False)
        self.de1 = nn.Sequential(
            spnn.Conv3d(128, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True),
            spnn.Conv3d(64, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        # Level 0
        self.up_0 = nn.Sequential(
            spnn.Conv3d(64, 32, 2, 2, transpose=True),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        self.lin_net_0 = spnn.Conv3d(64, 32, kernel_size=1, stride=1, bias=False)
        self.de0 = nn.Sequential(
            spnn.Conv3d(64, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True),
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        # Linear head
        self.output_layer = spnn.Conv3d(32, 20, kernel_size=1, stride=1, bias=True)



    def forward(self, features, indices):
        # input
        indices = indices.int()
        x = SparseTensor(features, indices)

        #-------------------------------- Input ----------------------------------------
        x_0 = self.input_conv(x)

        #-------------------------------- Encoder ----------------------------------------
        # Level 0
        x_en0 = self.en0(x_0)
        x_en0 = x_en0 + x_0

        # Level 1
        x_1 = self.down_0(x_en0)
        x_en1 = self.en1(x_1)
        x_en1 = x_en1 + x_1
    
        # Level 2
        x_2 = self.down_1(x_en1)
        x_en2 = self.en2(x_2)
        x_en2 = x_en2 + x_2

        # Level 3
        x_3 = self.down_2(x_en2)
        x_en3 = self.en3(x_3)
        x_en3 = x_en3 + x_3

        # Level 4
        x_4 = self.down_3(x_en3)
        x_en4 = self.en4(x_4)
        x_en4 = x_en4 + x_4

        # Level 5
        x_5 = self.down_4(x_en4)
        x_en5 = self.en5(x_5)
        x_en5 = x_en5 + x_5

        #-------------------------------- Middle ----------------------------------------
        x_m = self.down_5(x_en5)
        x_mid = self.mid(x_m)
        x_mid = x_mid + x_m

        #-------------------------------- Decoder ----------------------------------------
        # Level 5
        y_5 = self.up_5(x_mid)
        y_5 = torchsparse.cat([y_5, x_en5])
        y_de5 = self.de5(y_5)
        y_de5 = y_de5 + self.lin_net_5(y_5)

        # Level 4
        y_4 = self.up_4(y_de5)
        y_4 = torchsparse.cat([y_4, x_en4])
        y_de4 = self.de4(y_4)
        y_de4 = y_de4 + self.lin_net_4(y_4)

        # Level 3
        y_3 = self.up_3(y_de4)
        y_3 = torchsparse.cat([y_3, x_en3])
        y_de3 = self.de3(y_3)
        y_de3 = y_de3 + self.lin_net_3(y_3)

        # Level 2
        y_2 = self.up_2(y_de3)
        y_2 = torchsparse.cat([y_2, x_en2])
        y_de2 = self.de2(y_2)
        y_de2 = y_de2 + self.lin_net_2(y_2)

        # Level 1
        y_1 = self.up_1(y_de2)
        y_1 = torchsparse.cat([y_1, x_en1])
        y_de1 = self.de1(y_1)
        y_de1 = y_de1 + self.lin_net_1(y_1)

        # Level 0
        y_0 = self.up_0(y_de1)
        y_0 = torchsparse.cat([y_0, x_en0])
        y_de0 = self.de0(y_0)
        y_de0 = y_de0 + self.lin_net_0(y_0)
            
            
        #-------------------------------- output ----------------------------------------
        output = self.output_layer(y_de0)

        return output.F, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0


class Geo_branch(nn.Module):
    def __init__(self):
        
        super().__init__()

        #-------------------------------- Middle ----------------------------------------
        self.lin_mid_d = spnn.Conv3d(224, 32, kernel_size=1, stride=1, bias=False)
        
        self.lin_mid_i = nn.Linear(32, 32, bias=False)
        self.mid_geo = Intra_aggr_module(32, 32)
        self.lin_mid_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))

        #-------------------------------- Decoder ----------------------------------------
        # Level 5
        self.lin_de5_d = spnn.Conv3d(192, 32, kernel_size=1, stride=1, bias=False)
        
        self.cd_5 = Inter_fuse_module(32, 32)

        self.lin_de5_i = nn.Linear(96, 32, bias=False)
        self.de5_geo = Intra_aggr_module(32, 32)
        self.lin_de5_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))


        # Level 4
        self.lin_de4_d = spnn.Conv3d(160, 32, kernel_size=1, stride=1, bias=False)
        
        self.cd_4 = Inter_fuse_module(32, 32)

        self.lin_de4_i = nn.Linear(96, 32, bias=False)
        self.de4_geo = Intra_aggr_module(32, 32)
        self.lin_de4_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))


        # Level 3
        self.lin_de3_d = spnn.Conv3d(128, 32, kernel_size=1, stride=1, bias=False)
        
        self.cd_3 = Inter_fuse_module(32, 32)

        self.lin_de3_i = nn.Linear(96, 32, bias=False)
        self.de3_geo = Intra_aggr_module(32, 32)
        self.lin_de3_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))


        # Level 2
        self.lin_de2_d = spnn.Conv3d(96, 32, kernel_size=1, stride=1, bias=False)
        
        self.cd_2 = Inter_fuse_module(32, 32)

        self.lin_de2_i = nn.Linear(96, 32, bias=False)
        self.de2_geo = Intra_aggr_module(32, 32)
        self.lin_de2_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))


        # Level 1
        self.lin_de1_d = spnn.Conv3d(64, 32, kernel_size=1, stride=1, bias=False)
        
        self.cd_1 = Inter_fuse_module(32, 32)

        self.lin_de1_i = nn.Linear(96, 32, bias=False)
        self.de1_geo = Intra_aggr_module(32, 32)
        self.lin_de1_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))


        # Level 0
        self.lin_de0_d = spnn.Conv3d(32, 32, kernel_size=1, stride=1, bias=False)
        
        self.cd_0 = Inter_fuse_module(32, 32)

        self.lin_de0_i = nn.Linear(96, 32, bias=False)
        self.de0_geo = Intra_aggr_module(32, 32)
        self.lin_de0_o = nn.Sequential(
                            nn.Linear(32, 32, bias=False),
                            geo_norm.LayerNorm(32),
                            nn.ReLU(True))


        # Linear head
        self.output_layer = nn.Linear(32, 20, bias=True)
    


    def forward(self, mesh_data: list, traces: list, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0):

        # Geodesic
        mesh_l0 = mesh_data[0]
        mesh_l1 = mesh_data[1]
        mesh_l2 = mesh_data[2]
        mesh_l3 = mesh_data[3]
        mesh_l4 = mesh_data[4]
        mesh_l5 = mesh_data[5]
        mesh_mid = mesh_data[6]

        vertices_l0 = mesh_l0.pos
        vertices_l0 = PointTensor(None, vertices_l0)
        vertices_l1 = mesh_l1.pos
        vertices_l1 = PointTensor(None, vertices_l1)
        vertices_l2 = mesh_l2.pos
        vertices_l2 = PointTensor(None, vertices_l2)
        vertices_l3 = mesh_l3.pos
        vertices_l3 = PointTensor(None, vertices_l3)
        vertices_l4 = mesh_l4.pos
        vertices_l4 = PointTensor(None, vertices_l4)
        vertices_l5 = mesh_l5.pos
        vertices_l5 = PointTensor(None, vertices_l5)
        vertices_mid = mesh_mid.pos
        vertices_mid = PointTensor(None, vertices_mid)

        trace_l1 = traces[0]
        trace_l2 = traces[1]
        trace_l3 = traces[2]
        trace_l4 = traces[3]
        trace_l5 = traces[4]
        trace_mid = traces[5]

        #-------------------------------- Transition Middle ----------------------------------------
        x_mid_d = self.lin_mid_d(x_mid)
        z_mid_d = voxel_to_point(x_mid_d, vertices_mid)
        geo_mid_i = self.lin_mid_i(z_mid_d.F)
        mesh_mid.x = geo_mid_i
        geo_mid_o = self.mid_geo(mesh_mid) + geo_mid_i
        geo_mid_o = self.lin_mid_o(geo_mid_o)
        geo_de5 = geo_mid_o[trace_mid]

        #-------------------------------- Geodesic Decoder ----------------------------------------
        # Level 5
        y_de5_d = self.lin_de5_d(y_de5)
        z_de5 = voxel_to_point(y_de5_d, vertices_l5)
        # Cross domain attention
        geo_de5_at = self.cd_5(geo_de5, z_de5.F, mesh_l5.edge_index)
        geo_de5 = torch.cat([geo_de5, geo_de5_at, z_de5.F], 1)
        geo_de5_i = self.lin_de5_i(geo_de5)
        mesh_l5.x = geo_de5_i
        geo_de5_o = self.de5_geo(mesh_l5) + geo_de5_i
        geo_de5_o = self.lin_de5_o(geo_de5_o)
        geo_de4 = geo_de5_o[trace_l5]
        
        # Level 4
        y_de4_d = self.lin_de4_d(y_de4)
        z_de4 = voxel_to_point(y_de4_d, vertices_l4)
        # Cross domain attention
        geo_de4_at = self.cd_4(geo_de4, z_de4.F, mesh_l4.edge_index)
        geo_de4 = torch.cat([geo_de4, geo_de4_at, z_de4.F], 1)
        geo_de4_i = self.lin_de4_i(geo_de4)
        mesh_l4.x = geo_de4_i
        geo_de4_o = self.de4_geo(mesh_l4) + geo_de4_i
        geo_de4_o = self.lin_de4_o(geo_de4_o)
        geo_de3 = geo_de4_o[trace_l4]

        # Level 3
        y_de3_d = self.lin_de3_d(y_de3)
        z_de3 = voxel_to_point(y_de3_d, vertices_l3)
        # Cross domain attention
        geo_de3_at = self.cd_3(geo_de3, z_de3.F, mesh_l3.edge_index)
        geo_de3 = torch.cat([geo_de3, geo_de3_at, z_de3.F], 1)
        geo_de3_i = self.lin_de3_i(geo_de3)
        mesh_l3.x = geo_de3_i
        geo_de3_o = self.de3_geo(mesh_l3) + geo_de3_i
        geo_de3_o = self.lin_de3_o(geo_de3_o)
        geo_de2 = geo_de3_o[trace_l3]

        # Level 2
        y_de2_d = self.lin_de2_d(y_de2)
        z_de2 = voxel_to_point(y_de2_d, vertices_l2)
        # Cross domain attention
        geo_de2_at = self.cd_2(geo_de2, z_de2.F, mesh_l2.edge_index)
        geo_de2 = torch.cat([geo_de2, geo_de2_at, z_de2.F], 1)
        geo_de2_i = self.lin_de2_i(geo_de2)
        mesh_l2.x = geo_de2_i
        geo_de2_o = self.de2_geo(mesh_l2) + geo_de2_i
        geo_de2_o = self.lin_de2_o(geo_de2_o)
        geo_de1 = geo_de2_o[trace_l2]

        # Level 1
        y_de1_d = self.lin_de1_d(y_de1)
        z_de1 = voxel_to_point(y_de1_d, vertices_l1)
        # Cross domain attention
        geo_de1_at = self.cd_1(geo_de1, z_de1.F, mesh_l1.edge_index)
        geo_de1 = torch.cat([geo_de1, geo_de1_at, z_de1.F], 1)
        geo_de1_i = self.lin_de1_i(geo_de1)
        mesh_l1.x = geo_de1_i
        geo_de1_o = self.de1_geo(mesh_l1) + geo_de1_i
        geo_de1_o = self.lin_de1_o(geo_de1_o)
        geo_de0 = geo_de1_o[trace_l1]

        # Level 0
        y_de0_d = self.lin_de0_d(y_de0)
        z_de0 = voxel_to_point(y_de0_d, vertices_l0)
        # Cross domain attention
        geo_de0_at = self.cd_0(geo_de0, z_de0.F, mesh_l0.edge_index)
        geo_de0 = torch.cat([geo_de0, geo_de0_at, z_de0.F], 1)
        geo_de0_i = self.lin_de0_i(geo_de0)
        mesh_l0.x = geo_de0_i
        geo_de0_o = self.de0_geo(mesh_l0) + geo_de0_i
        geo_de0_o = self.lin_de0_o(geo_de0_o)
    
            
        #-------------------------------- output ----------------------------------------
        output = self.output_layer(geo_de0_o)

        return output


class VMNet(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.Euc_branch = Euc_branch()
        self.Geo_branch = Geo_branch()

    def forward(self, colors_v_b, coords_v_b, mesh_data: list, traces: list):

        out_euc, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0 = self.Euc_branch(colors_v_b, coords_v_b)
        out_geo = self.Geo_branch(mesh_data, traces, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0)
        
        return out_euc, out_geo
    

def model_fn(inference = False):

    def train_fn(model, batch):

        # Load data
        mesh_data = []
        mesh_data += [batch['mesh_l0_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l1_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l2_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l3_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l4_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l5_b'].to('cuda:0')]
        mesh_data += [batch['mesh_mid_b'].to('cuda:0')]  
        
        traces = []
        traces += [batch['trace_l1_b'].cuda()]
        traces += [batch['trace_l2_b'].cuda()]
        traces += [batch['trace_l3_b'].cuda()]
        traces += [batch['trace_l4_b'].cuda()]
        traces += [batch['trace_l5_b'].cuda()]
        traces += [batch['trace_mid_b'].cuda()]  

        coords_v_b = batch['coords_v_b'].cuda()
        colors_v_b = batch['colors_v_b'].cuda()
        labels_v_b = batch['labels_v_b'].cuda()
        labels_m_b = batch['labels_m_b'].cuda()

        # Forward
        out_euc, out_geo = model(colors_v_b, coords_v_b, mesh_data, traces)

        # Loss calculation
        loss = torch.nn.functional.cross_entropy(out_euc, labels_v_b, ignore_index=-100, reduction='mean') + \
                torch.nn.functional.cross_entropy(out_geo, labels_m_b, ignore_index=-100, reduction='mean')

        return loss


    def infer_fn(model, batch):
        
        # Load data
        mesh_data = []
        mesh_data += [batch['mesh_l0_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l1_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l2_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l3_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l4_b'].to('cuda:0')]
        mesh_data += [batch['mesh_l5_b'].to('cuda:0')]
        mesh_data += [batch['mesh_mid_b'].to('cuda:0')]  
        
        traces = []
        traces += [batch['trace_l1_b'].cuda()]
        traces += [batch['trace_l2_b'].cuda()]
        traces += [batch['trace_l3_b'].cuda()]
        traces += [batch['trace_l4_b'].cuda()]
        traces += [batch['trace_l5_b'].cuda()]
        traces += [batch['trace_mid_b'].cuda()]  

        coords_v_b = batch['coords_v_b'].cuda()
        colors_v_b = batch['colors_v_b'].cuda()                                      

        # Forward
        _, out_geo = model(colors_v_b, coords_v_b, mesh_data, traces)

        # reconstruct original vertices
        predictions = out_geo.cpu()
        trace_l0 = batch["trace_l0_b"]
        predictions = predictions[trace_l0]        

        return predictions


    if inference:
        fn = infer_fn
    else:
        fn = train_fn
    return fn

