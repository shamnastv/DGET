import copy

import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
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

    features, n, d = read_feature(dataset)
    label = read_label(dataset)
    adj, adj_norm = read_edgelist(dataset, n)

    adj = adj_norm
    return adj, adj_norm, features, label


def main():
    adj, adj_norm, features, label = load_data('cora')
    print(adj.shape)
    print(adj_norm.shape)
    print(features.shape)
    print(label)


if __name__ == '__main__':
    main()
