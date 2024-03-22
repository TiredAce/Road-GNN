import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from road_dataset.osm_graph import osm_graph_from_box

def load_road_dataset(north, south, east, west):
    graph = osm_graph_from_box(north, south, east, west)
    graph.init_graph()
    graph.merge_road()
    adj_matrix = graph.get_merged_adj_matrix()
    sim_adj_matrix = graph.get_similarity_of_adj_matrix()
    return adj_matrix, sim_adj_matrix

def load_dataset(dataset):
    
    name = ['Citeseer', 'Cora', 'Pubmed']

    if dataset not in name:
        raise("this dataset is not define.")
    else:
        data_dir = './dataset/' + dataset + '/'
        features = np.load(data_dir + 'feats.npy')
        adj = sp.load_npz(data_dir + 'adj.npz')
        label = np.load(data_dir + 'labels.npy')

        return features, adj, label