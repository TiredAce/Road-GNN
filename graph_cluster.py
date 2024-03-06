from osm_graph import osm_graph_from_box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import argparse
import os
import numpy as np
import negative_free_unsupervised_representation_learning_main as NF_n2n
import sklearn.cluster as clu
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import torch
import tools as tl

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

    parser.add_argument(
            '--alpha',
            type = float,
            default = 1.0)

    parser.add_argument(
            '--beta',
            type = float,
            default = 1.0)

    parser.add_argument(
            '-cl', 
            '--contrast-loss',
            type = str,
            default = 'distance')

    parser.add_argument(
            '-la',
            '--lam',
            type = float,
            default = 0.0)


    parser.add_argument(
            '-t',
            '--temperature',
            type = float,
            default = 1.0)

    parser.add_argument(
            '-r',
            '--num-run',
            type = int,
            default = 1)

    parser.add_argument(
            '-bs',
            '--batch-size',
            type = int,
            default = 2048) #65536

    parser.add_argument(
            '-zi',
            '--ZCA-iteration',
            type = int,
            default = 4)

    parser.add_argument(
            '-zt',
            '--ZCA-temp',
            type = float,
            default = 0.05)

    parser.add_argument(
            '-zl',
            '--ZCA-lam',
            type = float,
            default = 0.0)

    parser.add_argument(
            '-bn', 
            '--batch-normalization',
            type = str,
            default = 'Schur_ZCA') # None, BN, ZCA, Schur_ZCA

    parser.add_argument(
            '-pe',
            '--pretext-num-epochs',
            type = int,
            default = 30)

    parser.add_argument(
            '-pu',
            '--pretext-hidden-units',
            nargs='+', 
            type=int,
            default = [512])

    parser.add_argument(
            '-pt',
            '--pretext-output-units',
            type = int,
            default = 256)

    parser.add_argument(
            '-po',
            '--pretext-dropout',
            type = float,
            default = 0.)

    parser.add_argument(
            '-pL',
            '--pretext-L2',
            type = float,
            default = 0.)

    parser.add_argument(
            '-pl',
            '--pretext-learning-rate',
            type = float,
            default = 0.001)
    
    parser.add_argument(
            '--clusters',
            type = int,
            default = 10)
    
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    north, south, east, west = args.north, args.south, args.east, args.west

    # Data Preprocess
    graph = osm_graph_from_box(north, south, east, west)
    graph.init_graph()
    graph.merge_road()

    adj_matrix = graph.get_merged_adj_matrix()
    sim_adj_matrix = graph.get_similarity_of_adj_matrix()

    anchor_indexes, augmented_indexes = tl.get_positive_samples(sim_adj_matrix, threshold = 0.5)

    # Calculating the representation.
    # representation_adj = NF_n2n.get_representations(args, adj_matrix).cpu().detach().numpy()
    representation_sim = NF_n2n.get_representations(args, adj_matrix, anchor_indexes, augmented_indexes).cpu().detach().numpy()

    # torch.save(representation_adj, 'representation_adj.pt')
    torch.save(representation_sim, 'representation_sim.pt')

    # exit()

    # cluster_1 = clu.KMeans(args.clusters)

    # cluster_labels_1 = cluster_1.fit_predict(representation_adj)

    cluster_2 = clu.KMeans(40, n_init = 10)

    cluster_labels_2 = cluster_2.fit_predict(representation_sim)

    silhouette_avg_ = silhouette_score(representation_sim, cluster_labels_2)
    print("Silhouette Score:", silhouette_avg_)

    for i in range(40):
        graph.draw_merge_graph(cluster_labels_2, i)