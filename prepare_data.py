import argparse
import csv
import glob
import os
import shutil
from collections import defaultdict
from typing import List
import numpy as np
import open3d
import torch
from sklearn.neighbors import BallTree
import multiprocessing as mp
from functools import partial
from plyfile import PlyData

import sys
sys.path.append("..")


remapper = np.ones(150) * (-100)
for i, x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x] = i


def clear_folder(folder: str):
    """create temporary empty folder.
    If it already exists, all containing files will be removed.

    Arguments:
        folder {[str]} -- Path to the empty folder
    """
    if not os.path.exists(os.path.dirname(folder)):
        os.makedirs(os.path.dirname(folder))

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path) and ('.csv' in file_path or '.ply' in file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def edges_from_faces(faces):
    """extract non-repeating edges from triangular faces

    Args:
        faces ([N, 3]): list of triangles

    Returns:
        [dict]: each group contains the one ring neighbors of the key
    """
    edges = defaultdict(set)
    for i in range(len(faces)):
        edges[faces[i, 0]].update(faces[i, (1, 2)])
        edges[faces[i, 1]].update(faces[i, (0, 2)])
        edges[faces[i, 2]].update(faces[i, (0, 1)])

    edge_list = []

    for vertex_id in range(len(edges)):
        connected_vertices = edges[vertex_id]
        edge_list.append(list(connected_vertices))

    return edge_list


def csv2npy(in_file_path, old_vertices, new_vertices):
    """compute the trace between two levels of meshes simplified by QEM
    """
    old_ball_tree = BallTree(old_vertices[:, :3])
    new_ball_tree = BallTree(new_vertices[:, :3])

    new2old = {}
    old_nodes_set = set()
    new_nodes_set = set()

    with open(in_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            new_coords = [float(r) for r in row[:3]]
            num_traces = len(row) // 3 - 1

            new_id = new_ball_tree.query(
                np.array(new_coords).reshape(1, -1), k=1)[1].flatten()[0]
            if not new2old.get(new_id):
                new2old[new_id] = []
            else:
                raise ValueError('GRAPH LEVEL GENERATION ERROR')

            new_nodes_set.add(new_id)

            for i in range(num_traces):
                old_coords = [float(r) for r in row[3+3*i:6+3*i]]
                old_id = old_ball_tree.query(
                    np.array(old_coords).reshape(1, -1), k=1)[1].flatten()[0]
                new2old[new_id].append(old_id)
                old_nodes_set.add(old_id)

    all_old_ids = list(
        set([i for i in range(old_vertices.shape[0])]) - old_nodes_set)
    if np.array(old_vertices[all_old_ids]).shape[0] != 0:
        new_ids = new_ball_tree.query(
            np.array(old_vertices[all_old_ids]), k=1)[1].flatten()
        new_nodes_set.update(new_ids.tolist())
        old_nodes_set.update(all_old_ids)

        for id, new_id in enumerate(new_ids):
            if not new2old.get(new_id):
                new2old[new_id] = []
            new2old[new_id].append(all_old_ids[id])

    assert new_vertices.shape[0] == len(new_nodes_set)
    assert old_vertices.shape[0] == len(old_nodes_set)

    reverse_trace = np.empty((len(old_nodes_set)), dtype=np.int32)
    reverse_trace.fill(-1)

    for new_id, old_ids in new2old.items():
        for old_id in old_ids:
            assert reverse_trace[old_id] == -1
            reverse_trace[old_id] = new_id

    return reverse_trace


def vertex_clustering(curr_file_path, voxel_size, old_vertices):
    # Call vertex clustering func
    os.system(f"trimesh_clustering "
                f"{curr_file_path} {curr_file_path} -s {voxel_size} > /dev/null")
    
    mesh_l = open3d.io.read_triangle_mesh(curr_file_path)
    if not mesh_l.has_vertices():
        raise ValueError('no vertices left')

    vertices_l = np.asarray(mesh_l.vertices)
    edge_dict_l = edges_from_faces(np.asarray(mesh_l.triangles))

    # Save edges of this level
    edges_l = []
    for key, group in enumerate(edge_dict_l):
        for elem in group:
            edges_l.append([key, elem])

    # Save trace to the previous level
    vh_ball_tree = BallTree(vertices_l[:, :3])
    traces_l = vh_ball_tree.query(
        np.asarray(old_vertices[:, :3]), k=1)[1].flatten()

    return vertices_l, edges_l, traces_l


def quadric_error_metric(curr_file_path, ratio, old_vertices):
    # Call tridecimator func
    os.system(f"tridecimator "
                f"{curr_file_path} {curr_file_path} {ratio} -On -C > /dev/null")

    mesh_l = open3d.io.read_triangle_mesh(curr_file_path)
    if not mesh_l.has_vertices():
        raise ValueError('no vertices left')
    
    vertices_l = np.asarray(mesh_l.vertices)
    edge_dict_l = edges_from_faces(np.asarray(mesh_l.triangles))
    
    # Save edges of this level
    edges_l = []
    for key, group in enumerate(edge_dict_l):
        for elem in group:
            edges_l.append([key, elem])

    # Save trace to the previous level
    traces_l = csv2npy(curr_file_path.replace(
        '.ply', '.csv'), old_vertices=old_vertices, new_vertices=vertices_l)

    return vertices_l, edges_l, traces_l


def process_frame(file_path: str, level_params: list, out_path: str, test_split: bool):
    
    print("Processing " + file_path)

    # Load mesh
    original_mesh = open3d.io.read_triangle_mesh(file_path)
    original_vertices = np.asarray(original_mesh.vertices)
    original_vertices = original_vertices - original_vertices.mean(0)
    original_mesh.vertices = open3d.utility.Vector3dVector(original_vertices)
    
    # Load label
    if test_split:
        pass
    else:
        labels_file_path = file_path.replace('.ply', '.labels.ply')
        vertex_labels = np.asarray(PlyData.read(labels_file_path)['vertex']['label'])
        vertex_labels = remapper[vertex_labels]

    # create train/val dataset
    subfolder = f"{file_path.split('/')[-2]}"
    curr_dir = f"{out_path}{subfolder}"
    print("curr_dir: " + curr_dir)

    clear_folder(f"{curr_dir}/")

    # Containers
    vertices = []
    edges = []
    traces = []

    # Original data
    curr_mesh = original_mesh
    curr_vertices = np.asarray(curr_mesh.vertices)
    vertices.append(curr_vertices)

    # put current mesh in the working directory
    curr_mesh_path = f"{curr_dir}/curr_mesh.ply"
    open3d.io.write_triangle_mesh(curr_mesh_path, curr_mesh)

    # Mesh simplification
    for level in range(len(level_params)):
        if level_params[level] < 1:
            vertices_l, edges_l, traces_l = \
                vertex_clustering(curr_mesh_path, level_params[level],
                                        old_vertices=vertices[-1])
        else:
            vertices_l, edges_l, traces_l = \
                quadric_error_metric(curr_mesh_path, int(level_params[level]),
                                        old_vertices=vertices[-1])

        vertices.append(vertices_l)
        traces.append(traces_l)
        edges.append(np.array(edges_l)) 


    # Calculate labels for mesh of level 0
    if test_split:
        pass
    else:
        ball_tree_ori = BallTree(original_vertices[:, :3])
        _, ind = ball_tree_ori.query(vertices[1], k=1)
        labels_l0 = vertex_labels[ind.flatten()]

    # Clear current folder
    clear_folder(f"{curr_dir}/")

    # Save to pt data
    pt_data = {}
    pt_data['vertices'] = [torch.from_numpy(vertices[i]).float() for i in range(len(vertices))]
    pt_data['edges'] = [torch.from_numpy(edges[i]).long() for i in range(len(edges))]
    pt_data['traces'] = [torch.from_numpy(x).long() for x in traces]
    pt_data['colors'] = torch.from_numpy(np.asarray(original_mesh.vertex_colors) * 2 - 1).float()
    if test_split:
        pass
    else:
        pt_data['labels'] = torch.from_numpy(vertex_labels).long()
        pt_data['labels_l0'] = torch.from_numpy(labels_l0).long()
    torch.save(pt_data, f"{curr_dir}.pt")

    # Delete empty folder
    shutil.rmtree(f"{curr_dir}/")
    print(file_path + " completed.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='VMNet Data Preparation')
    parser.add_argument('--test_split', dest='test_split', action='store_true')
    parser.set_defaults(test_split=False)
    parser.add_argument('--considered_rooms_path', default=None, type=str, required=True,
                        help='data split file (default: None)')
    parser.add_argument('--in_path', default=None, type=str, required=True,
                        help='path to scene data (default: None)')
    parser.add_argument('--out_path', default=None, type=str, required=True,
                        help='path for saving processed data (default: None)')
    args = parser.parse_args()
    
    considered_rooms_path = args.considered_rooms_path
    in_path = args.in_path
    out_path = args.out_path
    level_params = [0.02, 0.04, 30, 30, 30, 30, 30]

    print('considered_rooms_path:' + considered_rooms_path)
    print('in_path:' + in_path)
    print('out_path:' + out_path)
    print(level_params)

    # Load file paths
    with open(considered_rooms_path, 'r') as f:
        considered_rooms = f.read().splitlines()
    file_paths = sorted([x for x in glob.glob(f"{in_path}/*/*.ply")
                        if 'clean_2.ply' in x and x.split('/')[-1].rsplit('_', 3)[0] in considered_rooms])

    # Partial function
    process_frame_p = partial(process_frame, level_params=level_params, out_path=out_path, test_split=args.test_split)
    # multi-processing
    pf_pool = mp.Pool(processes=12)
    pf_pool.map(process_frame_p, file_paths)
    pf_pool.close()
    pf_pool.join()