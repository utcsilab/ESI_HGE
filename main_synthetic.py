from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import warnings
import os
import json
from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, show_graph_with_labels
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from torch import nn
import manifolds
import json
from synthetic import SyntheticDataset

from geoopt.manifolds.poincare.math import  dist

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return 'cuda:'+str(np.argmax(memory_available))


device = torch.cuda.is_available()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=123456789, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2, help='Number of units in hidden layer 2.')
parser.add_argument('--gamma', type=float, default=1, help='coefficient for the information term')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='synthetic', help='type of dataset.')
parser.add_argument('--device', type=str, default=get_freer_gpu() if device else 'cpu')
parser.add_argument('--noise_dim', type=int, default=1)
parser.add_argument('--K', type=int, default=18)
parser.add_argument('--J', type=int, default=3)
parser.add_argument('--c', type=float, default=1., help='constant of curvature')
parser.add_argument('--warmup_de', type=float, default=30.)
parser.add_argument('--final_latent', type=str, default=True)
parser.add_argument('--start_latent_display', type=int, default=0)
parser.add_argument('--reduced_latent_size', type=int, default=1000)
parser.add_argument('--latent_display_show', type=int, default=50)
parser.add_argument('--latent_animation', type=str, default=False)
parser.add_argument('--syn_dim', type=list, default=[64, 64])
parser.add_argument('--syn_depth', type=int, default=6)
parser.add_argument('--new_generation', type=bool, default=True)
args = parser.parse_args()

warnings.filterwarnings('ignore')


class ExpZero(nn.Module):
    def __init__(self, manifold):
        super(ExpZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)


class LogZero(nn.Module):
    def __init__(self, manifold):
        super(LogZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.logmap0(input)

class Discriminator(nn.Module):
    def __init__(self, feature_dim=2, z_dim=2):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.feature_dim, 1000),
            nn.ReLU(False),
            nn.Linear(1000, 400),
            nn.ReLU(False),
            nn.Linear(400, 100),
            nn.ReLU(False),
            nn.Linear(100, 1),

        )

    def forward(self, x, z):
        x = x.view(-1, 64*64)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()


def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to(args.device)
    perm_z = z[perm]
    return perm_z

def gae_for(args):
    torch.manual_seed(args.seed + 1)
    print("Using {} dataset".format(args.dataset_str))
    if args.dataset_str in ['cora', 'citeseer', 'pubmed']:
        adj, features, labels = load_data(args.dataset_str)
        print(adj.shape, features.shape)
        exit()

    elif args.dataset_str == 'synthetic':
        if args.new_generation:
            dict_adj, adj_array, features = SyntheticDataset(args.syn_dim, args.syn_depth).__getitem__()
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(dict_adj))
            features = (255 - features) / 255.
            features = torch.Tensor(features)
        else:
            with open('adj_dict.json', 'r') as fp:
                dict_adj = json.load(fp)

            adj_dict = {}
            for a in dict_adj:
                adj_dict[int(a)] = dict_adj[a]

            adj_array, features = np.load('adjacancy.npy'), np.load('features.npy')
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj_dict))
            features = torch.Tensor(features)
    else:
        raise ValueError('not exist!!!')

    features = features.to(args.device).unsqueeze(1)
    n_nodes, _, feat_dim_hight, feat_dim_length = features.shape

        # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj=adj,
                                                                                                           args=args)
    adj = adj_train

        # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_norm = adj_norm.to(args.device)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())
    adj_orig_tile = adj_label.unsqueeze(2).repeat(1, 1, args.K)
    adj_orig_tile = adj_orig_tile.to(args.device)

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).float().to(
            device=args.device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    psi_input_dim = args.noise_dim + feat_dim_hight + feat_dim_length
    logv_input_dim = feat_dim_hight + feat_dim_length

    model = GCNModelVAE(psi_input_dim, logv_input_dim, args.hidden1, args.hidden2, args.dropout, args.K, args.J, args.noise_dim, args.device, args.c).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    D = Discriminator(feature_dim=64*64, z_dim=args.hidden2).to(args.device)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0005)
    manifold = getattr(manifolds, 'PoincareBall')(args.hidden2, args.c)

    latent_img = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mapper = LogZero(manifold)

    for epoch in range(args.epochs):
        warm_up = torch.min(torch.FloatTensor([epoch/args.warmup_de, 1])).to(args.device)

        t = time.time()
        model.train()

        reconstruct_iw, log_prior_iw, log_H_iw, psi_iw_vec, psi_iw = model(features, adj_norm)
        hidden_emb = psi_iw[:, 1, :].data.contiguous().cpu().numpy()
        z_vec = mapper(psi_iw)

        loss1 = loss_function(reconstructed_iw=reconstruct_iw, log_prior_iw=log_prior_iw, log_H_iw=log_H_iw,
                             adj_orig_tile=adj_orig_tile, nodes=n_nodes, K=args.K, pos_weight=pos_weight, norm=norm,
                             warm_up=warm_up, device=args.device)
        for i in range(int(args.K/2)):
            z = z_vec[:, i]
            D_xz = D(features, z)
            z_perm = permute_dims(z)
            D_x_z = D(features, z_perm)
            output_ = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))
            if i == 0:
                output = output_.unsqueeze(0)
            else:
                output = torch.cat((output, output_.unsqueeze(0)), dim=0)

        Info_xz = output.mean()

        loss = loss1 + args.gamma + Info_xz

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_D.zero_grad()
        Info_xz.backward()

        optimizer.step()
        optimizer_D.step()

        cur_loss = loss.item()
        print('Epoch:', '%04d --->   ' % (epoch + 1), 'training_loss = {:.5f}   '.format(cur_loss),
              'time = {:.5f}   '.format(time.time() - t))

        writer.add_scalar('Loss/train_loss', cur_loss, epoch)

        #print("Optimization Finished!")
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(adj_array.shape[0]):
            for j in range(adj_array.shape[0]):
                if adj_array[i, j] == 1:
                    x_vals = [hidden_emb[i, 0], hidden_emb[j, 0]]
                    y_vals = [hidden_emb[i, 1], hidden_emb[j, 1]]
                    ax.plot(x_vals, y_vals, color='blue', linewidth=0.8)

        for i in range(adj_array.shape[0]):

            ax.scatter(hidden_emb[i, 0],
            hidden_emb[i, 1],
            cmap='jet', c='black', edgecolors=None, s=20)


        ax.set_xlim(
            [-1 / np.sqrt(args.c) - 0.2 * (1 / np.sqrt(args.c)), 1 / np.sqrt(args.c) + 0.2 * (1 / np.sqrt(args.c))])
        ax.set_ylim(
            [-1 / np.sqrt(args.c) - 0.2 * (1 / np.sqrt(args.c)), 1 / np.sqrt(args.c) + 0.2 * (1 / np.sqrt(args.c))])
        patch = plt.Circle((0, 0), radius=1 / np.sqrt(args.c), color='black', fill=False)
        ax.add_patch(patch)
        if epoch > 60:
            plt.show()
        fig.savefig('moreeps/reduced_latent_more_{}.pdf'.format(epoch), format='pdf', dpi=500)
        #hidden_emb = torch.from_numpy(hidden_emb)
        #A = torch.zeros(hidden_emb.shape[0], hidden_emb.shape[0])
        #for i in range(hidden_emb.shape[0]):
        #    for j in range(hidden_emb.shape[0]):
        #        A[i, j] = dist(hidden_emb[i], hidden_emb[j], c=args.c)

        #print(A)
        #exit()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(adj_array.shape[0]):
            for j in range(adj_array.shape[0]):
                if adj_array[i, j] == 1:
                    x_vals = [hidden_emb[i, 0], hidden_emb[j, 0]]
                    y_vals = [hidden_emb[i, 1], hidden_emb[j, 1]]
                    ax.plot(x_vals, y_vals, color='blue', linewidth=0.8)

        for i in range(adj_array.shape[0]):

            ax.scatter(hidden_emb[i, 0],
            hidden_emb[i, 1],
            cmap='jet', c='black', edgecolors=None, s=20)


        for i in range(adj_array.shape[0]):
            ax.annotate(str(i), (hidden_emb[i, 0], hidden_emb[i, 1]))

        ax.set_xlim(
            [-1 / np.sqrt(args.c) - 0.2 * (1 / np.sqrt(args.c)), 1 / np.sqrt(args.c) + 0.2 * (1 / np.sqrt(args.c))])
        ax.set_ylim(
            [-1 / np.sqrt(args.c) - 0.2 * (1 / np.sqrt(args.c)), 1 / np.sqrt(args.c) + 0.2 * (1 / np.sqrt(args.c))])
        patch = plt.Circle((0, 0), radius=1 / np.sqrt(args.c), color='black', fill=False)
        ax.add_patch(patch)
        fig.savefig('moreeps/reduced_latent_{}.pdf'.format(epoch), format='pdf', dpi=500)


if __name__ == '__main__':
    print('New_Experiment', 'c:{}'.format(args.c), 'K:{}'.format(args.K), 'J:{}'.format(args.J),
          'learning_rate:{}'.format(args.lr),
          'warm_up:{}'.format(args.warmup_de), 'hidden1:{}'.format(args.hidden1), 'hidden2:{}'.format(args.hidden2),
          'droput:{}'.format(args.dropout))
    tensorboard_file_name = '___Run_ID___' + '__c' + str(args.c) + '__K' + str(args.K) + '__J' + str(args.K) + \
                            '__lr' + str(args.lr) + '__warm_up' + str(args.warmup_de) + '__hidden1_' + str(
        args.hidden1) + \
                            '__hidden2_' + str(args.hidden2) + '__dropout' + str(args.dropout)
    writer = SummaryWriter(log_dir='./logs', filename_suffix=tensorboard_file_name)
    gae_for(args)
