import copy
import pickle

import scipy.sparse as sp
import numpy as np


def read_data(dataset):
    # datatype = 'coauthorship/'
    datatype = 'cocitation/'
    edgelist_path = './hyper_datasets/' + datatype + dataset + '/' + 'hypergraph.pickle'
    label_path = './hyper_datasets/' + datatype + dataset + '/' + 'labels.pickle'
    content_path = './hyper_datasets/' + datatype + dataset + '/' + 'features.pickle'
    with open(edgelist_path, 'rb') as handle:
        hyper_graph = pickle.load(handle)
    with open(label_path, 'rb') as handle:
        labelset = pickle.load(handle)
    with open(content_path, 'rb') as handle:
        contentset = pickle.load(handle).todense()
    hyper_edge_list = list(hyper_graph.values())

    labelset = np.array(labelset)
    no_of_nodes = labelset.shape[0]
    no_of_edges = len(hyper_edge_list)
    hyper_incidence_matrix = np.zeros((no_of_nodes, no_of_edges), dtype=float)
    for i in range(len(hyper_edge_list)):
        for node in hyper_edge_list[i]:
            hyper_incidence_matrix[node][i] = 1

    # print(hyper_incidence_matrix[1700:1900, -1])

    adj_norm = normalize(hyper_incidence_matrix)

    adj = copy.deepcopy(adj_norm)
    # adj[adj > 0] = 1

    # adj_norm = adj_norm + np.identity(no_of_nodes)
    # adj_norm = normalize2(adj_norm)

    return adj, adj_norm, contentset, labelset

    # hyper_incidence_matrix_norm = copy.deepcopy(hyper_incidence_matrix) + sp.identity(no_of_nodes)
    # hyper_incidence_matrix_norm = normalize_h(hyper_incidence_matrix_norm).todense()
    # return hyper_incidence_matrix.todense(), hyper_incidence_matrix_norm, contentset, labelset
    # hyper_incidence_matrix_norm = normalize_h(hyper_incidence_matrix)
    # return hyper_incidence_matrix, hyper_incidence_matrix_norm, contentset, labelset


def normalize2(mx):
    """Row-normalize sparse matrix"""
    mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.todense()


def normalize_h(mx):
    """Row-normalize sparse matrix"""
    mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    colsum = np.array(mx.sum(0))

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    c_inv_sqrt = np.power(colsum, -0.5).flatten()
    c_inv_sqrt[np.isinf(c_inv_sqrt)] = 0.
    c_mat_inv_sqrt = sp.diags(c_inv_sqrt)

    mx = r_mat_inv_sqrt.dot(mx).dot(c_mat_inv_sqrt)

    return mx.todense()


def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    colsum = np.array(mx.sum(0))

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)

    mx = r_mat_inv_sqrt.dot(mx).dot(c_mat_inv).dot(mx.transpose()).dot(r_mat_inv_sqrt)

    return mx.todense()


def main():
    adj, adj_norm, features, label = read_data('cora')
    print(adj)
    print(adj_norm)
    print(features)
    print(label[0:200])


if __name__ == '__main__':
    main()
