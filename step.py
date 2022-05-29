import numpy as np
import csv
import  pandas as pd
import  matplotlib.pyplot as plt
from numpy import *
import pickle
from math import radians, cos, sin, asin, sqrt, atan2
import math
from dgl import DGLGraph
from numpy import unravel_index
from scipy import sparse
import torch
import sklearn
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g.to(device)
        self.fc = nn.Linear(in_dim, out_dim, bias=False).to(device)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False).to(device)

    def edge_attention(self, edges):
        inter = torch.diag(torch.mm(edges.src['z'].to(device), torch.transpose(edges.dst['z'].to(device), 0, 1)))
        norm_z1 = torch.diag(
            torch.sqrt(torch.mm(edges.src['z'].to(device), torch.transpose(edges.src['z'].to(device), 0, 1))))
        norm_z2 = torch.diag(
            torch.sqrt(torch.mm(edges.dst['z'].to(device), torch.transpose(edges.dst['z'].to(device), 0, 1))))

        a = inter

        return {'e': a.to(device)}

    def message_func(self, edges):
        return {'z': edges.src['z'].to(device), 'e': edges.data['e'].to(device)}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1).to(device)
        h = torch.mm(alpha, nodes.mailbox['z'][0, :, :]).to(device)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h.to(device)).to(device)
        self.g.ndata['z'] = z.to(device)
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h').to(device)


def edge_attention(self, edges):
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e': F.leaky_relu(a)}


def reduce_func(self, nodes):
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h': h}


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_locs, num_heads, pred_horizon, input_attn_dim):
        super(GAT, self).__init__()

        self.Wq = torch.eye(in_dim)
        self.ig = g.to(device)

        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim1 * num_heads, hidden_dim2, 1)

        self.num_locs = num_locs

        self.pred_horizon = pred_horizon

        self.gru = nn.GRUCell(hidden_dim2, gru_dim)

        self.nn_res1 = nn.Linear(gru_dim + 2, 2 * pred_horizon)
        self.nn_res2 = nn.Linear(gru_dim + 2, 2)
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim

    def input_attention(self, edges):

        dot_p = torch.diag(torch.mm(edges.src['iz'].to(device), torch.transpose(edges.dst['iz'].to(device), 0, 1)))
        norm_h1 = torch.diag(
            torch.sqrt(torch.mm(edges.src['iz'].to(device), torch.transpose(edges.src['iz'].to(device), 0, 1))))
        norm_h2 = torch.diag(
            torch.sqrt(torch.mm(edges.dst['iz'].to(device), torch.transpose(edges.dst['iz'].to(device), 0, 1))))

        iz2 = (dot_p / norm_h1 / norm_h2) ** 4

        return {'e': iz2.to(device)}

    def input_message_func(self, edges):
        return {'iz': edges.src['iz'].to(device), 'e': edges.data['e'].to(device)}

    def input_reduce_func(self, nodes):
        alpha = nodes.mailbox['e'].to(device)
        ih = torch.mm(alpha, nodes.mailbox['iz'][0, :, :]).to(device)
        return {'ih': ih}

    def forward(self, h, N, I, R, S, It, Rt):
        T = h.size(0)
        N = N.squeeze()

        hx = torch.randn(1, self.gru_dim).to(device)

        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        self.alpha_scaled = []
        self.beta_scaled = []

        for each_step in range(T):
            iz = h[each_step, :]
            self.ig.ndata['iz'] = iz.to(device)
            self.ig.apply_edges(self.input_attention)
            self.ig.update_all(self.input_message_func, self.input_reduce_func)
            ih = self.ig.ndata.pop('ih').to(device)

            cur_h = self.layer1(ih)
            cur_h = F.relu(cur_h)
            cur_h = self.layer2(cur_h)
            cur_h = torch.max(F.relu(cur_h), 0)[0].reshape(1, self.hidden_dim2)
            hx = self.gru(cur_h, hx)

            new_hx = torch.cat((hx, It[each_step].reshape(1, 1), Rt[each_step].reshape(1, 1)), dim=1)

            pred_res = self.nn_res1(new_hx).squeeze()
            alpha, beta = self.nn_res2(new_hx).squeeze()
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            alpha = torch.sigmoid(alpha)
            beta = torch.sigmoid(beta)
            self.alpha_scaled.append(alpha)
            self.beta_scaled.append(beta)

            I_idx = [2 * i for i in range(self.pred_horizon)]
            R_idx = [2 * i + 1 for i in range(self.pred_horizon)]
            new_I.append(pred_res[I_idx])
            new_R.append(pred_res[R_idx])

            for i in range(self.pred_horizon):
                last_I = I[each_step] if i == 0 else last_I + dI.detach()
                last_R = R[each_step] if i == 0 else last_R + dR.detach()
                last_S = S[each_step] if i == 0 else N - last_I - last_R
                dI = alpha * last_I * (last_S / N) - beta * last_I
                dR = beta * last_I  # 1
                phy_I.append(dI)
                phy_R.append(dR)

        new_I = torch.stack(new_I).to(device).squeeze()
        new_R = torch.stack(new_R).to(device).squeeze()
        phy_I = torch.stack(phy_I).to(device).squeeze()
        phy_R = torch.stack(phy_R).to(device).squeeze()

        self.alpha_list = torch.stack(self.alpha_list).to(device).squeeze()
        self.beta_list = torch.stack(self.beta_list).to(device).squeeze()
        self.alpha_scaled = torch.stack(self.alpha_scaled).to(device).squeeze()
        self.beta_scaled = torch.stack(self.beta_scaled).to(device).squeeze()
        return new_I, new_R, phy_I, phy_R




features = pd.read_csv("dataset_all.csv",low_memory=False)
content = []
with open( "adjmatrix_all.csv", encoding='UTF-8') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        content.append(row)
distance1 = []
for i in range(1,len(content)):
    distance1.append(content[i])
distance1 = np.array(distance1)
#print(distance1.shape)
distance = [[]for i in range(51)]
for i in range(len(distance1)):
    for j in range(len(distance1[0])):
        distance[i].append(float(distance1[i][j]))

confirmed = np.array(features)[:,2]
active = np.array(features)[:,3]
death = np.array(features)[:,4]
recovered = np.array(features)[:,5]
other_feat = np.array(features)[:,4:] # includes the active, confirmed # cases

confirmed_cases = torch.from_numpy(np.reshape(confirmed,(51, 153),order='C').astype('float64'))
death_cases = torch.from_numpy(np.reshape(death,(51, 153),order='C').astype('float64'))
recovered_cases = torch.from_numpy(np.reshape(recovered,(51, 153),order='C').astype('float64'))
active_cases = torch.from_numpy(np.reshape(active,(51, 153),order='C').astype('float64'))

feat_tensor = np.reshape(np.array(other_feat),(51,153,other_feat.shape[1]),order='C')
feat_tensor = torch.from_numpy(feat_tensor.astype('float64'))
print("Feature tensor is of size ", feat_tensor.shape)

popn = np.array(features['popu']).astype('double')
pop_data=torch.tensor(popn).view(51,153)

W_sparse=sparse.coo_matrix(distance)

values = W_sparse.data
indices = np.vstack((W_sparse.row, W_sparse.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = W_sparse.shape

Adj=torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

row_sum = torch.sum(Adj, dim=1)
row_sum[row_sum == 0] = 1

invD_sqrt = torch.diag(torch.sqrt(row_sum.pow_(-1)))
Adj_norm = torch.mm(invD_sqrt, (torch.mm(Adj, invD_sqrt)))
Adj_norm = (Adj_norm - torch.diag(torch.diag(Adj_norm))) + torch.eye(Adj_norm.shape[0])

sel=np.unique(features['state'])

g=DGLGraph(sparse.csr_matrix(Adj_norm.data.numpy()))

selected_counties = sel
Adj = Adj_norm
popn = pop_data


d_confirmed = []
d_death = []
d_recovered = []
d_active = []

for i in range(confirmed_cases.size(0)):
    i_confirmed = [0]
    i_death = [0]
    i_recovered = [0]
    i_active = [0]
    for j in range(1, confirmed_cases.size(1)):
        i_confirmed.append(confirmed_cases[i, j] - confirmed_cases[i, j - 1])
        i_death.append(death_cases[i, j] - death_cases[i, j - 1])
        i_recovered.append(recovered_cases[i, j] - recovered_cases[i, j-1])
        i_active.append(active_cases[i, j])
    d_confirmed.append(i_confirmed)
    d_death.append(i_death)
    d_recovered.append(i_recovered)
    d_active.append(i_active)
#print(confirmed_cases.shape)
#print(len(d_confirmed[0]))
d_confirmed = torch.tensor(d_confirmed).to(device)
d_death = torch.tensor(d_death).to(device)
d_recovered = torch.tensor(d_recovered).to(device)
d_active = torch.tensor(d_active).to(device)
slot_len = 6
pred_horizon = 60
step_size = 5
mid_time = (pred_horizon//2)+1

nhid1 = 400
nhid2 = 200
gru_dim = 100
num_heads = 3
lr = 1e-2     #学习率
r = 1

num_epochs = 50

N, T, D = feat_tensor.shape
nfeat = slot_len * D
nclass = pred_horizon * 3
attn_dim = int(nfeat * 1.5)

num_pred_time = len(range(slot_len, T))
error=open('error_512_abs_used_version.txt','a')
error_30=open('error_512_abs_used_version_f30.txt','a')
num=open('confirmed_512_abs_used_version.csv','a')
num_30=open('confirmed_512_abs_used_version_f30.csv','a')

z=1
for i in range(1):
    num_val_pred_time = 60
    num_test_pred_time = 5+5*z
    num_train_pred_time = num_pred_time-num_val_pred_time - num_test_pred_time
    print(num_train_pred_time)
    test_pred_time=np.arange(T-num_test_pred_time,T)
    val_pred_time=np.arange(test_pred_time[0]-num_val_pred_time,test_pred_time[0])
    train_pred_time = np.arange(slot_len,val_pred_time[0]+pred_horizon-1)

    train_input_time = np.arange(train_pred_time[-1]+1-pred_horizon)
    val_input_time = np.arange(val_pred_time[0]-slot_len,val_pred_time[-1]+1-pred_horizon)
    a = np.array([123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152])

    test_input_time = np.array(a[0:5+5*z])

    # train, val, test split along the spatial dimension

    num_train = int(1 * N)
    num_val = int(0 * N)
    num_test = N - num_train - num_val

    # shuffle_order = np.random.permutation(N)
    shuffle_order = np.arange(N)

    train_locs = np.arange(N)[shuffle_order[:num_train]]
    val_locs = np.arange(N)[shuffle_order[num_train:num_train + num_val]]
    test_locs = np.arange(N)[shuffle_order[num_train + num_val:]]

    # data splitting

    W_train=Adj[train_locs,:][:,train_locs].to(device)
    train_features=feat_tensor[train_locs,:,:][:,train_input_time,:][:].to(device)
    train_labels_active = active_cases[train_locs,:][:,train_pred_time].to(device)
    train_d_active = d_active[train_locs,:][:,train_pred_time].to(device)
    train_labels_confirmed = confirmed_cases[train_locs,:][:,train_pred_time].to(device)
    train_labels_recovered = recovered_cases[train_locs,:][:,train_pred_time].to(device)
    train_d_recovered = d_recovered[train_locs,:][:,train_pred_time].to(device)
    train_popn=popn[train_locs,:][:].to(device)

    # For validation and testing, we use the full training data points as well

    val_set = np.concatenate((train_locs, val_locs), axis=0)

    W_val=Adj[val_set,:][:,val_set].to(device)
    val_features = feat_tensor[val_set,:,:][:, val_input_time,:][:].to(device)
    val_labels_active = active_cases[val_set,:][:,val_pred_time].to(device)
    val_d_active = d_active[val_set,:][:,val_pred_time].to(device)
    val_labels_confirmed = confirmed_cases[val_set,:][:,val_pred_time].to(device)
    val_popn=popn[val_set,:][:].to(device)
    val_d_recovered = d_recovered[val_set,:][:,val_pred_time].to(device)

    # For testing, we use the full training set as well for graph based inference

    test_set = np.concatenate((train_locs, test_locs), axis=0)

    W_test=Adj[test_set,:][:,test_set].to(device)
    test_features = feat_tensor[test_set, :,:][:,test_input_time].to(device)
    test_labels_active = d_active[test_set,:][:,test_pred_time].to(device)
    test_d_active =  active_cases[test_set,:][:,test_pred_time].to(device)
    test_labels_confirmed = confirmed_cases[test_set,:][:,test_pred_time].to(device)
    test_d_recovered = d_recovered[test_set,:][:,test_pred_time].to(device)
    content = [[]for i in range(num_epochs)]
    for i in range(51):
        target_loc=i
        print('-----------Start training for loc %d-----------' % i)
        num_train_steps = len(train_input_time) - slot_len + 1
        batch_feat = []
        batch_active = []
        batch_recover = []
        batch_I = []
        batch_R = []
        batch_S = []
        batch_It = []
        batch_Rt = []

        for t in range(0, num_train_steps, step_size):
            t_feat = train_features[:, t:t + slot_len, :].view(int(num_train), slot_len * D).float()
            t_active = train_d_active[:][target_loc, t:t + pred_horizon].float()
            t_recovered = train_d_recovered[:][target_loc, t:t + pred_horizon].float()
            last_I = active_cases[target_loc, train_pred_time[t] - 1].unsqueeze(-1).float().to(device)
            last_R = recovered_cases[target_loc, train_pred_time[t] - 1].unsqueeze(-1).float().to(device)
            last_S = popn[target_loc, 0].to(device).float().unsqueeze(-1) - last_I - last_R

            batch_It.append(d_active[target_loc, train_pred_time[t] - 1].float())
            batch_Rt.append(d_recovered[target_loc, train_pred_time[t] - 1].float())

            batch_feat.append(t_feat)
            batch_active.append(t_active)
            batch_recover.append(t_recovered)
            batch_I.append(last_I)
            batch_R.append(last_R)
            batch_S.append(last_S)

        batch_It = torch.stack(batch_It).to(device).squeeze()
        batch_Rt = torch.stack(batch_Rt).to(device).squeeze()
        batch_feat = torch.stack(batch_feat).to(device)
        batch_active = torch.stack(batch_active).to(device)
        batch_recover = torch.stack(batch_recover).to(device)
        batch_I = torch.stack(batch_I).to(device).squeeze()
        batch_R = torch.stack(batch_R).to(device).squeeze()
        batch_S = torch.stack(batch_S).to(device).squeeze()

        valid_feat = []
        valid_active = []
        valid_recover = []
        valid_I = []
        valid_R = []
        valid_S = []
        valid_It = []
        valid_Rt = []

        num_val_steps = val_features.shape[1] - slot_len + 1
        for t in range(0, num_val_steps, step_size):
            t_feat = val_features[:, t:t + slot_len, :].view(int(num_train), slot_len * D).float()
            t_active = val_d_active[:][target_loc, t:t + pred_horizon].float()
            t_active[0] = 300
            t_recovered = val_d_recovered[:][target_loc, t:t + pred_horizon].float()

            last_I = active_cases[target_loc, val_pred_time[t] - 1].unsqueeze(-1).float().to(device)
            last_R = recovered_cases[target_loc, val_pred_time[t] - 1].unsqueeze(-1).float().to(device)
            last_S = popn[target_loc, 0].to(device).float().unsqueeze(-1) - last_I - last_R

            valid_It.append(d_active[target_loc, val_pred_time[t] - 1].float())
            valid_Rt.append(d_recovered[target_loc, val_pred_time[t] - 1].float())

            valid_feat.append(t_feat)
            valid_active.append(t_active)
            valid_recover.append(t_recovered)
            valid_I.append(last_I)
            valid_R.append(last_R)
            valid_S.append(last_S)

        valid_It = torch.stack(valid_It).to(device).flatten()
        valid_Rt = torch.stack(valid_Rt).to(device).flatten()
        valid_feat = torch.stack(valid_feat).to(device)
        valid_active = torch.stack(valid_active).to(device)
        valid_I = torch.stack(valid_I).to(device).flatten()
        valid_R = torch.stack(valid_R).to(device).flatten()
        valid_S = torch.stack(valid_S).to(device).flatten()
        print(len(valid_I))
        # training the model
        model_name = selected_counties[target_loc]
        model = GAT(g, nfeat, nhid1, nhid2, gru_dim, N, num_heads, pred_horizon, attn_dim).to(device)
        model = model.float()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        criterion = nn.MSELoss(reduction='sum')
        if target_loc < 51:
            for epoch in range(num_epochs):

                model.train()
                optimizer.zero_grad()

                active_pred, recovered_pred, phy_active, phy_recover = model(batch_feat, popn[target_loc, 0].to(device).float(),
                                                                             batch_I, batch_R, batch_S, batch_It,
                                                                             batch_Rt)  # forward pass
                loss = F.mse_loss(active_pred, batch_active)
                loss.backward()
                optimizer.step()
                # Evaluate
                model.eval()
                with torch.no_grad():
                    active_val, recovered_val, vphy_active, vphy_recover = model(valid_feat,
                                                                                 popn[target_loc, 0].to(device).float(),
                                                                                 valid_I, valid_R, valid_S, valid_It,
                                                                                 valid_Rt)  # forward pass

        val = (valid_active.cpu().detach().numpy().tolist())[0]
        act = (active_val.cpu().detach().numpy().tolist())
        model.zero_grad()
        optimizer.zero_grad()