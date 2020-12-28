import numpy as np


def read_feature(filename):
    with open(filename) as f:
        n, d = f.readline().strip('\n').split(' ')
        n, d = int(n), int(d)
        features = []
        for line in f.readlines():
            features.append(list(map(float, line.strip('\n').split(' '))))

        features = np.array(features, dtype=np.float)
        return features, n, d


def read_label(filename):
    with open(filename) as f:
        lines = f.readlines()
        n = len(lines)
        label = np.zeros((n, ), dtype=np.int)
        for line in lines:
            i, lb = line.strip('\n').split(' ')
            label[int(i)] = int(lb)
    return label


def read_edgelist(filename, n):
    adj = np.zeros((n, n), dtype=np.float32)
    with open(filename) as f:
        for line in f.readlines():
            i, j = line.strip('\n').split(' ')
            i, j = int(i), int(j)
            adj[i, j] = 1
            adj[j, i] = 1

    adj_norm = adj + np.identity(n)
    adj_norm /= np.tile(np.mat(np.sum(adj_norm, axis=1)), (n, 1)).T

    return adj, adj_norm


def load_data(dataset):
    prefix = './datasets/' + dataset + '/' + dataset
    feature_file = prefix + '.feature'
    label_file = prefix + '.label'
    edge_file = prefix + '.edgelist'

    features, n, d = read_feature(feature_file)
    label = read_label(label_file)
    adj, adj_norm = read_edgelist(edge_file, n)

    return adj, adj_norm, features, label


def main():
    adj, adj_norm, features, label = load_data('citeseer')
    print(adj.shape)
    print(adj_norm.shape)
    print(features.shape)
    print(label)


if __name__ == '__main__':
    main()
