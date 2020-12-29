import copy
import pickle

import scipy.sparse as sp
import numpy as np


def read_data(dataset):
    edgelist_path = './hyper_datasets/' + dataset + '/' + 'hypergraph.pickle'
    label_path = './hyper_datasets/' + dataset + '/' + 'labels.pickle'
    content_path = './hyper_datasets/' + dataset + '/' + 'features.pickle'
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

    hyper_incidence_matrix = sp.coo_matrix(hyper_incidence_matrix)
    hyper_incidence_matrix = normalize(hyper_incidence_matrix).todense()

    adj = copy.deepcopy(hyper_incidence_matrix)
    adj[adj > 0] = 1

    return adj, hyper_incidence_matrix, contentset, labelset


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    colsum = np.array(mx.sum(0))

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)

    mx = r_mat_inv_sqrt.dot(mx).dot(c_mat_inv).dot(mx.transpose()).dot(r_mat_inv_sqrt)

    return mx


def main():
    adj, adj_norm, features, label = read_data('cora')
    print(adj.shape)
    print(adj_norm.shape)
    print(features.shape)
    print(label[0:200])


if __name__ == '__main__':
    main()
