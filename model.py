import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args, n, dim):
        super(Encoder, self).__init__()
        self.features_enc = nn.Sequential(nn.Linear(dim, args.hidden_dim_enc_feat), nn.Sigmoid())
        self.adj_enc = nn.Sequential(nn.Linear(n, args.hidden_dim_enc_adj), nn.Sigmoid())
        self.mu_layer = nn.Linear(args.hidden_dim_enc_feat + args.hidden_dim_enc_adj, args.hidden_dim)
        self.var_layer = nn.Linear(args.hidden_dim_enc_feat + args.hidden_dim_enc_adj, args.hidden_dim)

    def forward(self, adj, features):
        features_h = self.features_enc(features)
        adj_h = self.adj_enc(adj)
        h = torch.cat((adj_h, features_h), dim=1)
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
