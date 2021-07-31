from torch_geometric.utils import degree
import torch
from torch_sparse import coalesce
import math


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index,
                edge_attr=None,
                force_undirected=False,
                num_nodes=None,
                degrees=None,
                cutoff=10,
                alpha=1.):

    N = num_nodes
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    filter = (degrees > cutoff)[row].float()
    keep_probability = filter * torch.pow((degrees[row] + 1 - cutoff).float(), - alpha / math.log(cutoff+1, 2))

    keep_probability[(1-filter).byte()] = 1.

    mask = torch.bernoulli(keep_probability).byte()

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)],
            dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def edge_sampling(edge_index, cutoff=10, alpha=1.0):

    num_nodes = edge_index.max().item() + 1
    row, _ = edge_index
    node_degrees = degree(row, num_nodes, dtype=edge_index.dtype)

    new_edges, _ = dropout_adj(edge_index,
                                force_undirected=False,
                                num_nodes=num_nodes,
                                degrees=node_degrees,
                                cutoff=cutoff,
                                alpha=alpha)

    return new_edges
