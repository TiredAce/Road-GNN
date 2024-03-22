from road_dataset.osm_graph import osm_graph_from_box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import argparse
import os
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import model.model_N2N as NF_n2n
import sklearn.cluster as clu
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import torch
import tools as tl
import networkx as nx
from input_data import load_dataset, load_road_dataset
from model.model_N2N import unsupervised_learning_2

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', 
        '--dataset',
        type = str,
        default = 'Cora')
    
    parser.add_argument(
        '--use_road_dataset',
        action='store_true')
    
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
            default = 20)

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
            default = 100)

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
    
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    args.use_road_dataset = False

    if args.use_road_dataset:
        north, south, east, west = args.north, args.south, args.east, args.west
        adj, sim = load_road_dataset(north, south, east, west)
        anchor_indexes, augmented_indexes = tl.get_positive_samples(sim, threshold = 0.5)
        feat = adj

    else:
        feat, adj, label = load_dataset(args.dataset)
        anchor_indexes, augmented_indexes = tl.get_positive_samples(adj)

        label = np.argmax(label, axis = 1)

        evaluate_method = [normalized_mutual_info_score,
                       tl.variation_of_information,
                       tl.calculate_modularity,
                       tl.conductance,
                       tl.triangle_participation_ratio]
        
        print(1)

        eval = unsupervised_learning_2(feat,
                          adj,
                          anchor_indexes,
                          augmented_indexes,
                          args,
                          evaluate_method,
                          label)

        np.save('eval.npy', eval)
