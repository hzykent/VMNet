import numpy as np
import torch
import math
import torch.utils.data

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")

from torch_geometric.data import Data, Batch

from dataset.utils import edge_sampling


class Dataset:
    def __init__(self, train_files = None, infer_files = None, val = False,
                 scale = 50, batch_size = 8, full_scale = [4096, 4096, 4096], limit_numpoints = 800000, num_workers = 12):

        self.train_files = train_files
        self.infer_files = infer_files
        self.val = val
        
        self.scale = scale
        self.batch_size = batch_size
        self.full_scale = full_scale
        self.limit_numpoints = limit_numpoints
        self.num_workers = num_workers
        

    def train_data_loader(self):
        
        return torch.utils.data.DataLoader(
            list(range(len(self.train_files))),
            batch_size = self.batch_size,
            collate_fn = self.trainMerge,
            num_workers = self.num_workers,
            shuffle = True,
            drop_last = True,
            pin_memory = True
            )


    def infer_data_loader(self):
        
        infer_offsets=[0]
        if self.val:
            infer_labels=[]
        for _, infer_file in enumerate(self.infer_files):
            data = torch.load(infer_file)
            infer_offsets.append(infer_offsets[-1] + data['vertices'][0].shape[0])
            if self.val:
                infer_labels.append(data['labels'].numpy())
        self.infer_offsets = infer_offsets
        if self.val:
            self.infer_labels = np.hstack(infer_labels)
        
        return torch.utils.data.DataLoader(
            list(range(len(self.infer_files))),
            batch_size = self.batch_size,
            collate_fn = self.inferMerge,
            num_workers = self.num_workers,
            shuffle = True,
            pin_memory = True
            )   


    def trainMerge(self, tbl):
        
        # Batch data containers          
        mesh_l0_b = []
        mesh_l1_b = []
        mesh_l2_b = []
        mesh_l3_b = []
        mesh_l4_b = []
        mesh_l5_b = []
        mesh_mid_b = []
        
        trace_l1_b = [[np.array(-1)]]
        trace_l2_b = [[np.array(-1)]]
        trace_l3_b = [[np.array(-1)]]
        trace_l4_b = [[np.array(-1)]]
        trace_l5_b = [[np.array(-1)]]
        trace_mid_b = [[np.array(-1)]]
        
        coords_v_b = []         
        colors_v_b = []          
        labels_v_b = []
        labels_m_b = [] 

        # Process in batch
        batch_num_points = 0
        for idx, i in enumerate(tbl):
            
            # Load data
            data = torch.load(self.train_files[i])
            # vertices
            vertices_ori = data['vertices'][0] 
            vertices_l0 = data['vertices'][1]
            vertices_l1 = data['vertices'][2]
            vertices_l2 = data['vertices'][3]
            vertices_l3 = data['vertices'][4]
            vertices_l4 = data['vertices'][5]
            vertices_l5 = data['vertices'][6]
            vertices_mid = data['vertices'][7]
            # edges 
            edges_l0 = data['edges'][0]
            edges_l1 = data['edges'][1]
            edges_l2 = data['edges'][2]
            edges_l3 = data['edges'][3]
            edges_l4 = data['edges'][4]
            edges_l5 = data['edges'][5]
            edges_mid = data['edges'][6]
            # traces
            trace_l1 = data['traces'][1]
            trace_l2 = data['traces'][2]
            trace_l3 = data['traces'][3]
            trace_l4 = data['traces'][4]
            trace_l5 = data['traces'][5]
            trace_mid = data['traces'][6]
            # colors
            colors = data['colors']
            # labels
            labels = data['labels']
            labels_l0 = data['labels_l0']

            # Affine linear transformation
            trans_m = np.eye(3) + np.random.randn(3, 3) * 0.1
            trans_m[0][0] *= np.random.randint(0, 2) * 2 - 1
            trans_m *= self.scale
            theta = np.random.rand() * 2 * math.pi
            trans_m = np.matmul(trans_m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            
            vertices_ori = np.matmul(vertices_ori, trans_m)
            vertices_l0 = np.matmul(vertices_l0, trans_m)
            vertices_l1 = np.matmul(vertices_l1, trans_m)
            vertices_l2 = np.matmul(vertices_l2, trans_m)
            vertices_l3 = np.matmul(vertices_l3, trans_m)
            vertices_l4 = np.matmul(vertices_l4, trans_m)
            vertices_l5 = np.matmul(vertices_l5, trans_m)
            vertices_mid = np.matmul(vertices_mid, trans_m)

            # Random placement in the receptive field
            vertices_min = torch.min(vertices_ori, dim=0)[0].numpy()
            vertices_max = torch.max(vertices_ori, dim=0)[0].numpy()
            offset = -vertices_min + np.clip(self.full_scale - vertices_max + vertices_min - 0.001, 0, None) * np.random.rand(3) \
                + np.clip(self.full_scale - vertices_max + vertices_min + 0.001, None, 0) * np.random.rand(3)
            
            vertices_ori += offset
            vertices_l0 += offset
            vertices_l1 += offset
            vertices_l2 += offset
            vertices_l3 += offset
            vertices_l4 += offset
            vertices_l5 += offset
            vertices_mid += offset
            
            # Clip valid positions
            valid_idxs = (vertices_ori.min(1)[0] >= 0) * (vertices_ori.max(1)[0] < self.full_scale[0])
            assert sum(valid_idxs) == len(valid_idxs), 'input voxels are not valid'

            # Voxelization
            coords_v = vertices_ori.int()

            # Remove duplicate items
            _, unique_idxs = np.unique(coords_v, axis=0, return_index=True)
            coords_v = coords_v[unique_idxs]
            colors_v = colors[unique_idxs]
            labels_v = labels[unique_idxs]
            
            # Check point number limit
            batch_num_points += coords_v.shape[0]
            if batch_num_points > self.limit_numpoints:
                break

            # Put into containers
            coords_v_b += [torch.cat([coords_v, torch.IntTensor(coords_v.shape[0], 1).fill_(idx)], 1)]
            colors_v_b += [colors_v + torch.randn(3) * 0.1]
            labels_v_b += [labels_v]
            labels_m_b += [labels_l0]
            
            # Mesh batch
            vertices_l0 = torch.cat([vertices_l0.float(), torch.FloatTensor(vertices_l0.shape[0], 1).fill_(idx)], 1)
            vertices_l1 = torch.cat([vertices_l1.float(), torch.FloatTensor(vertices_l1.shape[0], 1).fill_(idx)], 1)
            vertices_l2 = torch.cat([vertices_l2.float(), torch.FloatTensor(vertices_l2.shape[0], 1).fill_(idx)], 1)
            vertices_l3 = torch.cat([vertices_l3.float(), torch.FloatTensor(vertices_l3.shape[0], 1).fill_(idx)], 1)
            vertices_l4 = torch.cat([vertices_l4.float(), torch.FloatTensor(vertices_l4.shape[0], 1).fill_(idx)], 1)
            vertices_l5 = torch.cat([vertices_l5.float(), torch.FloatTensor(vertices_l5.shape[0], 1).fill_(idx)], 1)
            vertices_mid = torch.cat([vertices_mid.float(), torch.FloatTensor(vertices_mid.shape[0], 1).fill_(idx)], 1)

            edges_l0 = edges_l0.t().contiguous()
            edges_l0 = edge_sampling(edges_l0, cutoff=4)
            mesh_l0_data = Data(pos=vertices_l0, edge_index=edges_l0)
            mesh_l0_b += [mesh_l0_data]

            edges_l1 = edges_l1.t().contiguous()
            edges_l1 = edge_sampling(edges_l1, cutoff=4)
            mesh_l1_data = Data(pos=vertices_l1, edge_index=edges_l1)
            mesh_l1_b += [mesh_l1_data]

            edges_l2 = edges_l2.t().contiguous()
            edges_l2 = edge_sampling(edges_l2, cutoff=4)
            mesh_l2_data = Data(pos=vertices_l2, edge_index=edges_l2)
            mesh_l2_b += [mesh_l2_data]

            edges_l3 = edges_l3.t().contiguous()
            edges_l3 = edge_sampling(edges_l3, cutoff=4)
            mesh_l3_data = Data(pos=vertices_l3, edge_index=edges_l3)
            mesh_l3_b += [mesh_l3_data]

            edges_l4 = edges_l4.t().contiguous()
            edges_l4 = edge_sampling(edges_l4, cutoff=4)
            mesh_l4_data = Data(pos=vertices_l4, edge_index=edges_l4)
            mesh_l4_b += [mesh_l4_data]

            edges_l5 = edges_l5.t().contiguous()
            edges_l5 = edge_sampling(edges_l5, cutoff=4)
            mesh_l5_data = Data(pos=vertices_l5, edge_index=edges_l5)
            mesh_l5_b += [mesh_l5_data]
            
            edges_mid = edges_mid.t().contiguous()
            edges_mid = edge_sampling(edges_mid, cutoff=4)
            mesh_mid_data = Data(pos=vertices_mid, edge_index=edges_mid)
            mesh_mid_b += [mesh_mid_data]

            # batch of traces
            trace_offset_l1 = max(trace_l1_b[-1]) + 1
            trace_l1_b += [trace_l1 + trace_offset_l1]
            trace_offset_l2 = max(trace_l2_b[-1]) + 1
            trace_l2_b += [trace_l2 + trace_offset_l2]
            trace_offset_l3 = max(trace_l3_b[-1]) + 1
            trace_l3_b += [trace_l3 + trace_offset_l3]
            trace_offset_l4 = max(trace_l4_b[-1]) + 1
            trace_l4_b += [trace_l4 + trace_offset_l4]
            trace_offset_l5 = max(trace_l5_b[-1]) + 1
            trace_l5_b += [trace_l5 + trace_offset_l5]
            trace_offset_mid = max(trace_mid_b[-1]) + 1
            trace_mid_b += [trace_mid + trace_offset_mid]


        # Construct batches
        coords_v_b = torch.cat(coords_v_b, 0)
        colors_v_b = torch.cat(colors_v_b, 0)
        labels_v_b = torch.cat(labels_v_b, 0)
        labels_m_b = torch.cat(labels_m_b, 0)
            
        mesh_l0_b = Batch.from_data_list(mesh_l0_b)
        mesh_l1_b = Batch.from_data_list(mesh_l1_b)
        mesh_l2_b = Batch.from_data_list(mesh_l2_b)
        mesh_l3_b = Batch.from_data_list(mesh_l3_b)
        mesh_l4_b = Batch.from_data_list(mesh_l4_b)
        mesh_l5_b = Batch.from_data_list(mesh_l5_b)
        mesh_mid_b = Batch.from_data_list(mesh_mid_b)

        trace_l1_b = torch.cat(trace_l1_b[1:], 0).long()
        trace_l2_b = torch.cat(trace_l2_b[1:], 0).long()
        trace_l3_b = torch.cat(trace_l3_b[1:], 0).long()
        trace_l4_b = torch.cat(trace_l4_b[1:], 0).long()
        trace_l5_b = torch.cat(trace_l5_b[1:], 0).long()
        trace_mid_b = torch.cat(trace_mid_b[1:], 0).long()

        return {'coords_v_b': coords_v_b, 'colors_v_b': colors_v_b, 'labels_v_b': labels_v_b, 'labels_m_b': labels_m_b,  
            'mesh_l0_b': mesh_l0_b, 'mesh_l1_b': mesh_l1_b, 'mesh_l2_b': mesh_l2_b, 'mesh_l3_b': mesh_l3_b, 'mesh_l4_b': mesh_l4_b, 'mesh_l5_b': mesh_l5_b, 'mesh_mid_b': mesh_mid_b,
            'trace_l1_b': trace_l1_b, 'trace_l2_b': trace_l2_b, 'trace_l3_b': trace_l3_b, 'trace_l4_b': trace_l4_b, 'trace_l5_b': trace_l5_b, 'trace_mid_b': trace_mid_b}
            
    
    def inferMerge(self, tbl):
    
        # Batch data containers          
        mesh_l0_b = []
        mesh_l1_b = []
        mesh_l2_b = []
        mesh_l3_b = []
        mesh_l4_b = []
        mesh_l5_b = []
        mesh_mid_b = []
        
        trace_l0_b = [[np.array(-1)]]
        trace_l1_b = [[np.array(-1)]]
        trace_l2_b = [[np.array(-1)]]
        trace_l3_b = [[np.array(-1)]]
        trace_l4_b = [[np.array(-1)]]
        trace_l5_b = [[np.array(-1)]]
        trace_mid_b = [[np.array(-1)]]
        
        coords_v_b = []         
        colors_v_b = []          
        
        point_ids = []

        # Process in batch    
        for idx, i in enumerate(tbl):
            
            # Load data
            data = torch.load(self.infer_files[i])
            # vertices
            vertices_ori = data['vertices'][0] 
            vertices_l0 = data['vertices'][1]
            vertices_l1 = data['vertices'][2]
            vertices_l2 = data['vertices'][3]
            vertices_l3 = data['vertices'][4]
            vertices_l4 = data['vertices'][5]
            vertices_l5 = data['vertices'][6]
            vertices_mid = data['vertices'][7]
            # edges 
            edges_l0 = data['edges'][0]
            edges_l1 = data['edges'][1]
            edges_l2 = data['edges'][2]
            edges_l3 = data['edges'][3]
            edges_l4 = data['edges'][4]
            edges_l5 = data['edges'][5]
            edges_mid = data['edges'][6]
            # traces
            trace_l0 = data['traces'][0]
            trace_l1 = data['traces'][1]
            trace_l2 = data['traces'][2]
            trace_l3 = data['traces'][3]
            trace_l4 = data['traces'][4]
            trace_l5 = data['traces'][5]
            trace_mid = data['traces'][6]
            # colors
            colors = data['colors']    
        
            # Affine linear transformation
            trans_m = np.eye(3)
            trans_m[0][0] *= np.random.randint(0, 2) * 2 - 1
            trans_m *= self.scale
            theta = np.random.rand() * 2 * math.pi
            trans_m = np.matmul(trans_m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            
            vertices_ori = np.matmul(vertices_ori, trans_m)
            vertices_l0 = np.matmul(vertices_l0, trans_m)
            vertices_l1 = np.matmul(vertices_l1, trans_m)
            vertices_l2 = np.matmul(vertices_l2, trans_m)
            vertices_l3 = np.matmul(vertices_l3, trans_m)
            vertices_l4 = np.matmul(vertices_l4, trans_m)
            vertices_l5 = np.matmul(vertices_l5, trans_m)
            vertices_mid = np.matmul(vertices_mid, trans_m)
            
            # Random placement in the receptive field
            vertices_min = torch.min(vertices_ori, dim=0)[0].numpy()
            vertices_max = torch.max(vertices_ori, dim=0)[0].numpy()
            offset = -vertices_min + np.clip(self.full_scale - vertices_max + vertices_min - 0.001, 0, None) * np.random.rand(3) \
                + np.clip(self.full_scale - vertices_max + vertices_min + 0.001, None, 0) * np.random.rand(3)
            
            vertices_ori += offset
            vertices_l0 += offset
            vertices_l1 += offset
            vertices_l2 += offset
            vertices_l3 += offset
            vertices_l4 += offset
            vertices_l5 += offset
            vertices_mid += offset

            # Clip valid positions
            valid_idxs = (vertices_ori.min(1)[0] >= 0) * (vertices_ori.max(1)[0] < self.full_scale[0])
            assert sum(valid_idxs) == len(valid_idxs), 'input voxels are not valid'

            # Voxelization
            coords_v = vertices_ori.int()

            # Remove duplicate items
            _, unique_idxs = np.unique(coords_v, axis=0, return_index=True)
            coords_v = coords_v[unique_idxs]
            colors_v = colors[unique_idxs] 

            # Put into containers
            coords_v_b += [torch.cat([coords_v, torch.IntTensor(coords_v.shape[0], 1).fill_(idx)], 1)]
            colors_v_b += [colors_v]

            # Mesh batch
            vertices_l0 = torch.cat([vertices_l0.float(), torch.FloatTensor(vertices_l0.shape[0], 1).fill_(idx)], 1)
            vertices_l1 = torch.cat([vertices_l1.float(), torch.FloatTensor(vertices_l1.shape[0], 1).fill_(idx)], 1)
            vertices_l2 = torch.cat([vertices_l2.float(), torch.FloatTensor(vertices_l2.shape[0], 1).fill_(idx)], 1)
            vertices_l3 = torch.cat([vertices_l3.float(), torch.FloatTensor(vertices_l3.shape[0], 1).fill_(idx)], 1)
            vertices_l4 = torch.cat([vertices_l4.float(), torch.FloatTensor(vertices_l4.shape[0], 1).fill_(idx)], 1)
            vertices_l5 = torch.cat([vertices_l5.float(), torch.FloatTensor(vertices_l5.shape[0], 1).fill_(idx)], 1)
            vertices_mid = torch.cat([vertices_mid.float(), torch.FloatTensor(vertices_mid.shape[0], 1).fill_(idx)], 1)
            
            edges_l0 = edges_l0.t().contiguous()
            mesh_l0_data = Data(pos=vertices_l0, edge_index=edges_l0)
            mesh_l0_b += [mesh_l0_data]

            edges_l1 = edges_l1.t().contiguous()
            mesh_l1_data = Data(pos=vertices_l1, edge_index=edges_l1)
            mesh_l1_b += [mesh_l1_data]

            edges_l2 = edges_l2.t().contiguous()
            mesh_l2_data = Data(pos=vertices_l2, edge_index=edges_l2)
            mesh_l2_b += [mesh_l2_data]

            edges_l3 = edges_l3.t().contiguous()
            mesh_l3_data = Data(pos=vertices_l3, edge_index=edges_l3)
            mesh_l3_b += [mesh_l3_data]

            edges_l4 = edges_l4.t().contiguous()
            mesh_l4_data = Data(pos=vertices_l4, edge_index=edges_l4)
            mesh_l4_b += [mesh_l4_data]

            edges_l5 = edges_l5.t().contiguous()
            mesh_l5_data = Data(pos=vertices_l5, edge_index=edges_l5)
            mesh_l5_b += [mesh_l5_data]
            
            edges_mid = edges_mid.t().contiguous()
            mesh_mid_data = Data(pos=vertices_mid, edge_index=edges_mid)
            mesh_mid_b += [mesh_mid_data]

            # batch of traces
            trace_offset_l0 = max(trace_l0_b[-1]) + 1
            trace_l0_b += [trace_l0 + trace_offset_l0]
            trace_offset_l1 = max(trace_l1_b[-1]) + 1
            trace_l1_b += [trace_l1 + trace_offset_l1]
            trace_offset_l2 = max(trace_l2_b[-1]) + 1
            trace_l2_b += [trace_l2 + trace_offset_l2]
            trace_offset_l3 = max(trace_l3_b[-1]) + 1
            trace_l3_b += [trace_l3 + trace_offset_l3]
            trace_offset_l4 = max(trace_l4_b[-1]) + 1
            trace_l4_b += [trace_l4 + trace_offset_l4]
            trace_offset_l5 = max(trace_l5_b[-1]) + 1
            trace_l5_b += [trace_l5 + trace_offset_l5]
            trace_offset_mid = max(trace_mid_b[-1]) + 1
            trace_mid_b += [trace_mid + trace_offset_mid]
                
            point_ids += [torch.nonzero(valid_idxs, as_tuple=True)[0] + self.infer_offsets[i]]


        # Construct batches
        coords_v_b = torch.cat(coords_v_b, 0)
        colors_v_b = torch.cat(colors_v_b, 0)

        mesh_l0_b = Batch.from_data_list(mesh_l0_b)
        mesh_l1_b = Batch.from_data_list(mesh_l1_b)
        mesh_l2_b = Batch.from_data_list(mesh_l2_b)
        mesh_l3_b = Batch.from_data_list(mesh_l3_b)
        mesh_l4_b = Batch.from_data_list(mesh_l4_b)
        mesh_l5_b = Batch.from_data_list(mesh_l5_b)
        mesh_mid_b = Batch.from_data_list(mesh_mid_b)

        trace_l0_b = torch.cat(trace_l0_b[1:], 0).long()
        trace_l1_b = torch.cat(trace_l1_b[1:], 0).long()
        trace_l2_b = torch.cat(trace_l2_b[1:], 0).long()
        trace_l3_b = torch.cat(trace_l3_b[1:], 0).long()
        trace_l4_b = torch.cat(trace_l4_b[1:], 0).long()
        trace_l5_b = torch.cat(trace_l5_b[1:], 0).long()
        trace_mid_b = torch.cat(trace_mid_b[1:], 0).long()

        point_ids = torch.cat(point_ids, 0)
        
        
        return {'coords_v_b': coords_v_b, 'colors_v_b': colors_v_b, 'point_ids': point_ids,
        'mesh_l0_b': mesh_l0_b, 'mesh_l1_b': mesh_l1_b, 'mesh_l2_b': mesh_l2_b, 'mesh_l3_b': mesh_l3_b, 'mesh_l4_b': mesh_l4_b, 'mesh_l5_b': mesh_l5_b, 'mesh_mid_b': mesh_mid_b,
        'trace_l0_b': trace_l0_b, 'trace_l1_b': trace_l1_b, 'trace_l2_b': trace_l2_b, 'trace_l3_b': trace_l3_b, 'trace_l4_b': trace_l4_b, 'trace_l5_b': trace_l5_b, 'trace_mid_b': trace_mid_b}
            