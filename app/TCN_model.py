import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.utils.parametrizations import weight_norm # pytorch update
from torch.nn.utils import weight_norm
import numpy as np
from scipy.special import expit

# encode amino acid seq as onehot vector
n_enc = 21

# This is a modified version of the TCN available at https://github.com/locuslab/TCN
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
            """Inputs have to have dimension (N, C_in, L_in)"""
            y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
            o_all = torch.stack([self.linear(y1[:, :, i]) for i in range(y1.shape[2])], dim=2) # return all outputs so we can select the correct index for the actual length of the non-padded sequence
            return o_all

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#def train(X, y, steps, batch_size, path):
#    model.train()
#
#    for i in range(steps):
#        if i%1000==100:
#            print(path+str(i)+".pt")
#            torch.save(model, path+str(i)+".pt")
#
#        select_idx = np.random.randint(0, len(X), batch_size) # random selection for now
#
#        X_tensor = torch.as_tensor(X[select_idx]).long()
#
#        y_tensor = torch.as_tensor(y[select_idx]).float()
#
#        X_select = F.one_hot(X_tensor, n_enc).permute(0,2,1).float().cuda() # TODO check how slow this is, better to minibatch encoding separately from model training?
#        y_select = y_tensor.view(-1, 1).cuda()
#        optimizer.zero_grad()
#        y_pred = model(X_select).view(-1, 1)
#        loss = criterion(y_pred, y_select)
#        loss.backward()
#        optimizer.step()
#        print(i, steps, round(loss.item(),3))
#        losslist.append(loss.item())

#def predict(X):
#    model.eval()
#    with torch.no_grad():
#        if torch.cuda.device_count() > 0:
#            X_enc = F.one_hot(X, 21).permute(0,2,1).float().cuda()
#            probs = expit(model(X_enc).cpu())
#            del X_enc
#            torch.cuda.empty_cache()
#        else:
#            X_enc = F.one_hot(X, 21).permute(0,2,1).float()
#            probs = expit(model(X_enc).cpu())
#    return probs
    
def tokenize_aa_seq(aa_seq):
    """Convert amino acid letters to integers. Could also use murphy's reduced aa alphabet"""
    table = {"L":1,
             "V":2,
             "I":3,
             "M":4,
             "C":5,
             "A":6,
             "G":7,
             "S":8,
             "T":9,
             "P":10,
             "F":11,
             "Y":12,
             "W":13,
             "E":14,
             "D":15,
             "N":16,
             "Q":17,
             "K":18,
             "R":19,
             "H":20,
             "X":1, # treat these as L, really shouldnt be in the data
             "B":1,
             "U":1,
             "Z":1,
             "J":1,
             "*":1,
             ".":1}
    # tokenized = torch.tensor([table[aa] for aa in aa_seq], dtype=int)
    tokenized = torch.tensor([table[aa] for aa in aa_seq])
    return tokenized
    