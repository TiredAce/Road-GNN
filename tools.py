import torch
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import networkx as nx
import torch.nn as nn
import scipy.sparse as sp
from networkx.algorithms.community.quality import modularity

def loss_dot_product(y_pred, true_pred, size_splits=None, temperature=1, bias=1e-8):
    numerator = torch.exp(torch.sum(y_pred * true_pred, dim=1, keepdim=True) / temperature)
    
    if size_splits is not None:
        y_pred_list = torch.split(y_pred, size_splits)
        den_list = []
        
        for y_pred in y_pred_list:
            E_1 = torch.matmul(y_pred, y_pred.t())
            den = torch.sum(torch.exp(E_1 / temperature), dim=1, keepdim=True)
            den_list.append(den)
           
        denominator = torch.cat(den_list, dim=0)
    else:
        E_1 = torch.matmul(y_pred, y_pred.t())
        denominator = torch.sum(torch.exp(E_1 / temperature), dim=1, keepdim=True)
    
    return -torch.mean(torch.log(numerator / (denominator + bias) + bias))

def loss_dot_product_v2(y_pred, true_pred, axis=1, temperature=1, bias=1e-8):
    loss = 2 - 2 * torch.sum(y_pred * true_pred, dim=axis, keepdim=True) \
        / (temperature * torch.norm(y_pred, dim=axis, keepdim=True) * torch.norm(true_pred, dim=axis, keepdim=True) + bias)
    return torch.mean(loss)

def loss_dot_product_v3(y_pred, true_pred, axis=1, temperature=1, bias=1e-8):
    numerator = torch.exp(torch.sum(y_pred * true_pred, dim=axis, keepdim=True) / temperature)
    
    if axis == 0:
        E_1 = torch.matmul(y_pred.t(), y_pred)
    else:
        E_1 = torch.matmul(y_pred, y_pred.t())
    
    denominator = torch.sum(torch.exp(E_1 / temperature), dim=axis, keepdim=True)
    
    return -torch.mean(torch.log(numerator / (denominator + bias) + bias))

def auto_correlation(y_pred, lam=0.005, bias=1e-8):
    D = y_pred.shape[-1]
    N = y_pred.shape[0]
    
    y_pred_mean = torch.mean(y_pred, dim=0, keepdim=True)
    y_pred = y_pred - y_pred_mean
    
    C = torch.matmul(y_pred.t(), y_pred) / N
    I = torch.eye(D)
    C_diff = torch.pow(C - I, 2)
    part_diag = torch.diag(C_diff)
    
    return torch.sum(part_diag) + torch.sum(lam * (C_diff - torch.diag(part_diag)))

def cross_correlation(y_pred, true_pred, lam=0.005, bias=1e-8):
    D = y_pred.shape[-1]
    N = y_pred.shape[0]
    
    y_pred_mean = torch.mean(y_pred, dim=0, keepdim=True)
    y_pred = y_pred - y_pred_mean
    
    true_pred_mean = torch.mean(true_pred, dim=0, keepdim=True)
    true_pred = true_pred - true_pred_mean

    C = torch.matmul(y_pred.t(), true_pred) / N
    I = torch.eye(D)
    C_diff = torch.pow(C - I, 2)
    part_diag = torch.diag(C_diff)
    
    return torch.sum(part_diag) + torch.sum(lam * (C_diff - torch.diag(part_diag)))


def Newton_ZCA_for_features(X, T=5, temp=10, epsilon=1e-8):
    """
    Apply Newton-ZCA whitening transform to input features.

    Args:
    X (torch.Tensor): Input features, a tensor of shape (m, d), where m is the number of samples and d is the dimensionality of each sample.
    T (int): Number of iterations for Newton's method. Defaults to 5.
    temp (float): Temperature parameter for whitening. Defaults to 10.
    epsilon (float): Small value to avoid singular matrix. Defaults to 1e-8.

    Returns:
    torch.Tensor: Transformed features after whitening.
    torch.Tensor: Mean of input features.
    torch.Tensor: Whitening matrix.
    """

    # Get dimensionality and number of samples
    d = X.shape[-1]
    m = X.shape[0]
    
    # Calculate mean of input features
    mean = torch.mean(X, dim=0, keepdim=True)
    
    # Center the input features
    X_mean = X - mean
    
    # Transpose of centered input features
    X_mean_T = X_mean.transpose(0, 1)
    
    # Compute covariance matrix with regularization
    C = torch.matmul(X_mean_T, X_mean) / m + epsilon * torch.eye(d)
    
    # Get diagonal part of the covariance matrix
    part_diag = torch.diag(C)
    
    # Compute trace of covariance matrix
    tr = torch.sum(part_diag)
    
    # Normalize covariance matrix
    C = C / tr
    
    # Initialize whitening matrix
    P = torch.eye(d)
    
    # Apply Newton's method for T iterations
    for _ in range(T):
        P = (3 * P - torch.matmul(P, torch.matmul(P, torch.matmul(P, C)))) / 2
        
    # Normalize the whitening matrix with temperature parameter and square root of trace
    C = P / (temp * torch.sqrt(tr))
    
    # Apply whitening transform to input features
    transformed_X = torch.matmul(X_mean, C)
    
    return transformed_X, mean, C

import torch

def Schur_Newton_ZCA_for_features(X, T=5, temp=10, epsilon=1e-8):
    """
    Apply Schur-Newton-ZCA whitening transform to input features.

    Args:
    X (torch.Tensor): Input features, a tensor of shape (m, d), where m is the number of samples and d is the dimensionality of each sample.
    T (int): Number of iterations for the Schur-Newton method. Defaults to 5.
    temp (float): Temperature parameter for whitening. Defaults to 10.
    epsilon (float): Small value to avoid singular matrix. Defaults to 1e-8.

    Returns:
    torch.Tensor: Transformed features after whitening.
    torch.Tensor: Mean of input features.
    torch.Tensor: Whitening matrix.
    """

    d = X.shape[-1]
    m = X.shape[0]
    I = torch.eye(d).to(X)
    
    # Calculate mean of input features
    mean = torch.mean(X, dim=0, keepdim=True)
    
    # Center the input features
    X_mean = X - mean
    
    # Transpose of centered input features
    X_mean_T = X_mean.transpose(0, 1)
    
    # Compute covariance matrix with regularization
    C = torch.matmul(X_mean_T, X_mean) / m + epsilon * I
    
    # Get diagonal part of the covariance matrix
    part_diag = torch.diag(C)
    
    # Compute trace of covariance matrix
    tr = torch.sum(part_diag)
    
    # Normalize covariance matrix
    C = C / tr
    
    # Initialize whitening matrix
    P = torch.eye(d).to(X)
    
    for _ in range(T):
        # Compute C_T
        C_T = (3 * I - C) / 2
        
        # Update whitening matrix
        P = torch.matmul(P, C_T)
        
        # Update covariance matrix
        C = torch.matmul(C_T, torch.matmul(C_T, C))
        
    # Normalize the whitening matrix with temperature parameter and square root of trace
    C = P / (temp * torch.sqrt(tr))
    
    # Apply whitening transform to input features
    transformed_X = torch.matmul(X_mean, C)
    
    return transformed_X, mean, C


def get_positive_samples(matrix, threshold = 0.5):
    anchor_indexes = []
    augmented_indexes = []
    
    if hasattr(matrix, 'scipy.sparse._csr.csr_matrix'):

        coo_matrix = matrix.tocoo()

        for row, col, val in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if row == col: continue
            if val >= threshold:
                anchor_indexes.append(row)
                augmented_indexes.append(col)
    else:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] >= threshold:
                    anchor_indexes.append(i)
                    augmented_indexes.append(j)

    anchor_indexes = np.array(anchor_indexes)
    augmented_indexes = np.array(augmented_indexes)

    return anchor_indexes, augmented_indexes

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


"""  ------Evaluation Metric-------      """

def entropy(labels):
    """计算熵"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def mutual_information(labels_true, labels_pred):
    """计算互信息"""
    entropy_true = entropy(labels_true)
    entropy_pred = entropy(labels_pred)
    
    joint_labels = list(zip(labels_true, labels_pred))
    unique_joint_labels, counts_joint_labels = np.unique(joint_labels, axis=0, return_counts=True)
    probabilities_joint_labels = counts_joint_labels / len(labels_true)
    
    mutual_info = np.sum(probabilities_joint_labels * np.log2(probabilities_joint_labels / (entropy_true * entropy_pred)))
    return mutual_info

def variation_of_information(labels_true, labels_pred):
    """计算VI"""
    mi = mutual_information(labels_true, labels_pred)
    vi = entropy(labels_true) + entropy(labels_pred) - 2 * mi
    return vi


def calculate_modularity(adjacency_matrix, community_labels):
    G = nx.from_numpy_array(adjacency_matrix)

    for node, label in enumerate(community_labels):
        G.nodes[node]['community'] = label

    modularity_score = modularity(G, [{node for node, data in G.nodes(data=True) if data['community'] == label} for label in set(community_labels)])

    return modularity_score

def conductance(adjacency_matrix, labels):
    """
    计算图的 Conductance
    
    参数：
    - adjacency_matrix: 邻接矩阵
    - labels: 节点标签或社区分配，是一个包含每个节点标签的列表
    """
    num_nodes = len(labels)
    total_volume = np.sum(np.sum(adjacency_matrix, axis=1))  # 总节点度数

    conductance = 0
    for label in set(labels):
        mask = np.array(labels) == label  # 标签为当前社区的节点掩码
        volume_internal = np.sum(np.sum(adjacency_matrix[mask, :], axis=1))  # 内部节点度数
        volume_external = total_volume - volume_internal  # 外部节点度数

        # 计算 Conductance
        if volume_internal + volume_external > 0:
            conductance += volume_external / (volume_internal + volume_external)

    conductance /= len(set(labels))  # 平均 Conductance
    return conductance


def triangle_participation_ratio(adjacency_matrix, labels):
    """
    计算图的 Triangle Participation Ratio (TPR)
    
    参数：
    - adjacency_matrix: 邻接矩阵
    - labels: 节点标签或社区分配，是一个包含每个节点标签的列表
    """
    num_nodes = len(labels)
    num_triangles_internal = 0
    num_triangles_external = 0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            for k in range(j + 1, num_nodes):
                if labels[i] == labels[j] == labels[k]:
                    if adjacency_matrix[i, j] == adjacency_matrix[j, k] == adjacency_matrix[i, k] == 1:
                        num_triangles_internal += 1
                else:
                    if adjacency_matrix[i, j] == adjacency_matrix[j, k] == adjacency_matrix[i, k] == 1:
                        num_triangles_external += 1

    tpr = num_triangles_internal / (num_triangles_internal + num_triangles_external)
    return tpr

"""  ------------------------------      """