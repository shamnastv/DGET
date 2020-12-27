import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=.5):
        super(GCN, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        h = self.linear1(x)
        h = torch.mm(adj, h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = torch.mm(adj, h)
        return torch.sigmoid(h)


class Encoder(nn.Module):
    def __init__(self, args, n, dim):
        super(Encoder, self).__init__()
        self.features_enc = nn.Sequential(nn.Linear(dim, args.hidden_dim_enc_feat), nn.Sigmoid())
        self.adj_enc = nn.Sequential(nn.Linear(n, args.hidden_dim_enc_adj), nn.Sigmoid())
        self.gcn_enc = GCN(dim, args.hidden_dim_enc_feat, args.hidden_dim_enc_feat)
        self.mu_layer = nn.Linear(args.hidden_dim_enc_feat * 2 + args.hidden_dim_enc_adj, args.hidden_dim)
        self.var_layer = nn.Linear(args.hidden_dim_enc_feat * 2 + args.hidden_dim_enc_adj, args.hidden_dim)
        # self.mu_layer = nn.Linear(args.hidden_dim_enc_feat + args.hidden_dim_enc_adj, args.hidden_dim)
        # self.var_layer = nn.Linear(args.hidden_dim_enc_feat + args.hidden_dim_enc_adj, args.hidden_dim)

    def forward(self, adj, features):
        features_h = self.features_enc(features)
        adj_h = self.adj_enc(adj)
        gcn_h = self.gcn_enc(adj, features)
        h = torch.cat((adj_h, features_h, gcn_h), dim=1)
        # h = torch.cat((adj_h, features_h), dim=1)
        mu = self.mu_layer(h)
        var = self.var_layer(h)

        return mu, var


class DGE(nn.Module):
    def __init__(self, args, n, dim):
        super(DGE, self).__init__()
        self.n = n
        self.encoder = Encoder(args, n, dim)
        self.feat_dec = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim_dec_feat),
                                      nn.Sigmoid(),
                                      nn.Linear(args.hidden_dim_dec_feat, dim))

    def forward(self, adj, features):
        mu, var = self.encoder(adj, features)
        z = mu + torch.exp(0.5 * var) * torch.randn_like(mu)
        adj_logit = torch.mm(z, z.t())
        feat_logits = self.feat_dec(z)
        return mu, var, adj_logit, feat_logits
