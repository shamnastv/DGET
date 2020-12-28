import copy
import pickle

import scipy.io as sio
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
    hyper_incidence_matrix = np.zeros((no_of_nodes, no_of_edges))
    for i in range(len(hyper_edge_list)):
        for node in hyper_edge_list[i]:
            hyper_incidence_matrix[node][i] = 1

    hyper_incidence_matrix = sp.coo_matrix(hyper_incidence_matrix)
    hyper_incidence_matrix = normalize(hyper_incidence_matrix).todense()

    return hyper_incidence_matrix, hyper_incidence_matrix, contentset, labelset


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    colsum = np.array(mx.sum(0))

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    c_inv_sqrt = np.power(colsum, -1).flatten()
    c_inv_sqrt[np.isinf(c_inv_sqrt)] = 0.
    c_mat_inv_sqrt = sp.diags(c_inv_sqrt)

    mx = r_mat_inv_sqrt.dot(mx).dot(c_mat_inv_sqrt).dot(mx.transpose()).dot(r_mat_inv_sqrt)

    return mx


def read_feature(dataset):
    filename = './datasets/' + dataset + '/' + dataset + '.feature'
    with open(filename) as f:
        if dataset == 'citeseer':
            n, d = f.readline().strip('\n').split(' ')
            n, d = int(n), int(d)
            features = []
            for line in f.readlines():
                features.append(list(map(float, line.strip('\n').split(' '))))

            features = np.array(features, dtype=np.float)
        else:
            features = []
            for line in f.readlines():
                features.append(list(map(float, line.strip('\n').split(','))))

            features = np.array(features, dtype=np.float)
            n, d = features.shape[0], features.shape[1]
        return features, n, d


def read_label(dataset):
    filename = './datasets/' + dataset + '/' + dataset + '.label'
    with open(filename) as f:
        lines = f.readlines()
        n = len(lines)
        label = np.zeros((n, ), dtype=np.int)
        for j, line in enumerate(lines):
            if dataset == 'citeseer':
                i, lb = line.strip('\n').split(' ')
            else:
                i, lb = j, line.strip()
            label[int(i)] = int(lb)
    return label


def read_edgelist(dataset, n):
    filename = './datasets/' + dataset + '/' + dataset + '.edgelist'
    adj = np.zeros((n, n), dtype=np.float32)
    with open(filename) as f:
        for line in f.readlines():
            i, j = line.strip('\n').split(' ')
            i, j = int(i), int(j)
            adj[i, j] = 1
            adj[j, i] = 1

    adj_norm = copy.deepcopy(adj)
    adj_norm = adj_norm + np.identity(n)
    # adj_norm /= np.tile(np.mat(np.sum(adj_norm, axis=1)), (n, 1)).T
    adj_norm = normalize(adj_norm)

    return adj, adj_norm


def load_data(dataset):

    if dataset == 'wiki':
        dataset = 'wiki'
        data = sio.loadmat('{}.mat'.format('./datasets/' + dataset + '/' + dataset))
        features = data['fea']
        if sp.issparse(features):
            features = features.todense()

        adj = data['W']
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        label = gnd[0, :]
        adj = sp.coo_matrix(adj)
        adj = adj.todense()
        n = adj.shape[0]
        adj_norm = copy.deepcopy(adj)
        adj_norm = adj_norm + np.identity(n)
        # adj_norm /= np.tile(np.mat(np.sum(adj_norm, axis=1)), (n, 1)).T
        adj_norm = normalize(adj_norm)
    else:

        features, n, d = read_feature(dataset)
        label = read_label(dataset)
        adj, adj_norm = read_edgelist(dataset, n)

    # adj = adj_norm
    return adj, adj_norm, features, label


def main():
    adj, adj_norm, features, label = read_data('cora')
    print(adj.shape)
    print(adj_norm.shape)
    print(features.shape)
    print(label[0:200])


if __name__ == '__main__':
    main()
