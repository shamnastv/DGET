import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mlxtend.evaluate import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from hyper_load_data import read_data
from hyper_model import DGE


def total_loss(args, mu, var, adj_logits, feat_logits, adj, features):
    kl_loss = -0.5 * torch.mean(torch.sum(torch.tensor(1).float().to(var.device) + var - mu ** 2 - torch.exp(var), dim=-1))

    # n = adj.shape[0]
    # pos = torch.sum(adj)
    # pos_weight = 2 * (n * n - pos) / pos
    # adj_loss = args.alpha * F.binary_cross_entropy_with_logits(adj_logits, adj, pos_weight=pos_weight)
    adj_loss = 100 * F.mse_loss(adj_logits, adj)

    feat_loss = 100 * F.mse_loss(feat_logits, features)
    print('Loss : KL:', kl_loss.detach().cpu().item(), ' ADJ :', adj_loss.detach().cpu().item()
          , 'FEAT :', feat_loss.detach().cpu().item())
    return kl_loss + adj_loss + feat_loss


def train(args, model, optimizer, adj, adj_norm, features):
    mu, var, adj_logits, feat_logits = model(adj_norm, features)

    loss = total_loss(args, mu, var, adj_logits, feat_logits, adj, features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(epoch, model, adj_norm, features, label):
    model.eval()
    p = .3
    with torch.no_grad():
        mu, _, _, _ = model(adj_norm, features)
    mu = mu.detach().cpu().numpy()
    train_X, test_X, train_y, test_y = train_test_split(mu, label, test_size=1.0 - p, random_state=1234)
    clf = LinearSVC()
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train_X, train_y)

    print('Epoch :', epoch)
    y_pred = clf.predict(train_X)
    macro = f1_score(y_pred, train_y, average='macro')
    micro = f1_score(y_pred, train_y, average='micro')
    accuracy = accuracy_score(train_y, y_pred)
    print('Train : Macro:{:.4f} Micro:{:.4f} Acc:{:.4f}'.format(macro, micro, accuracy))

    y_pred = clf.predict(test_X)
    macro = f1_score(y_pred, test_y, average='macro')
    micro = f1_score(y_pred, test_y, average='micro')
    accuracy = accuracy_score(test_y, y_pred)
    print('Test : Macro:{:.4f} Micro:{:.4f} Acc:{:.4f}'.format(macro, micro, accuracy))
    return macro, micro, accuracy


def main():
    parser = argparse.ArgumentParser(description='Torch for DGE')
    parser.add_argument("--dataset", type=str, default="cora", help="dataset name")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')

    parser.add_argument('--hidden_dim_enc_adj', type=int, default=512, help='hidden dimension')
    parser.add_argument('--hidden_dim_enc_feat', type=int, default=512, help='hidden dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--hidden_dim_dec_feat', type=int, default=128, help='hidden dimension')

    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout (default: 0.3)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.3)')

    parser.add_argument('--add_gcn', action="store_true", help='Whether add_gcn')
    parser.add_argument('--alpha', type=float, default=150.0, help='alpha')

    args = parser.parse_args()

    print(args, flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(0)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print('device : ', device, flush=True)

    adj, adj_norm, features, label = read_data(args.dataset)
    model = DGE(args, adj.shape[1], features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    features = torch.from_numpy(features).float().to(device)
    adj_norm = torch.from_numpy(adj_norm).float().to(device)
    adj = torch.from_numpy(adj).float().to(device)

    max_macro, max_micro, max_accuracy = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, adj, adj_norm, features)
        if epoch % 20 == 0:
            macro, micro, accuracy = test(epoch, model, adj_norm, features, label)
            max_macro = max(macro, max_macro)
            max_micro = max(micro, max_micro)
            max_accuracy = max(accuracy, max_accuracy)
            print('maxes :', max_macro, max_micro, max_accuracy)
            print('', flush=True)


if __name__ == '__main__':
    main()
