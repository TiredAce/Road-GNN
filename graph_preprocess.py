from osm_graph import osm_graph_from_box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import argparse
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', 
        '--dataset',
        type = str,
        default = 'Xian')
    
    parser.add_argument(
        '--dataset-dir',
        type = str,
        default='dataset')
    
    parser.add_argument(
        '-n', 
        '--north',
        type = float,
        default = 34.60)
    
    parser.add_argument(
        '-s', 
        '--south',
        type = float,
        default = 34.00)
    
    parser.add_argument(
        '-e', 
        '--east',
        type = float,
        default = 108.53)
    
    parser.add_argument(
        '-w', 
        '--west',
        type = float,
        default = 109.66)
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    # north, south, east, west = 22.573, 22.518, 114.1107, 114.024
    north, south, east, west = args.north, args.south, args.east, args.west

    graph = osm_graph_from_box(north, south, east, west)

    graph.init_graph()

    # graph.draw_orig_graph()

    graph.merge_road()


    # t = graph.draw_merge_graph()

    graph.get_merged_adj_matrix()

    sim_matrix = graph.get_similarity_of_adj_matrix()

    file_path = os.path.join(args.dataset_dir, args.dataset)
    os.makedirs(args.dataset_dir, exist_ok=True)
    save_npz(file_path, sim_matrix)
    