import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
from scipy.sparse import load_npz

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
        '--threshold',
        type = float,
        default = 0.5)

    args = parser.parse_args()

    return args



if __name__ == '__main__':

    args = get_args()

    file_path = os.path.join(args.dataset_dir, args.dataset + '.npz')

    sim_matrix = load_npz(file_path)

    anchor_indexes = []
    augmented_indexes = []

    coo_matrix = sim_matrix.tocoo()

    for row, col, val in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        if row == col: continue
        if val >= args.threshold:
            anchor_indexes.append(row)
            anchor_indexes.append(col)
            augmented_indexes.append(col)
            augmented_indexes.append(row)

    anchor_indexes = np.array(anchor_indexes)
    augmented_indexes = np.array(augmented_indexes)

    file_path = os.path.join(args.dataset_dir, args.dataset + '_anchor_indexes')
    np.save(file_path, anchor_indexes)

    file_path = os.path.join(args.dataset_dir, args.dataset + '_augmented_indexes')
    np.save(file_path, augmented_indexes)

