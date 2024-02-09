import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import collections
import numpy as np


class GLIMPSE(nn.Module):
    '''
    The encoder
    '''

    def __init__(self, window_size):
        '''
        ANP Encoder
        Can realize the CNP with identity attention
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(GLIMPSE, self).__init__()

        # self.positive_anchor_lstm = nn.LSTM(3,32)
        # self.patch_lstm = nn.LSTM(6,32)

        # self._is_positive_anchor = is_positive_anchor
        # self._is_patch = is_patch
        self.window_size = window_size

        num_ts = window_size

        latent_dim = 128
        # decoder_input = 64

        positive_anchor_patch_inp = 7

        positive_anchor_inp = 3
        patch_inp = 4
        # if not is_positive_anchor or not is_patch:
        #     decoder_input = 32

        self.positive_anchor_encoder = nn.Sequential(
            collections.OrderedDict([
                ("positive_anchorlin1", nn.Linear(positive_anchor_inp, latent_dim)),
                ("positive_anchorrelu1", nn.ReLU()),
                ("positive_anchorlin2", nn.Linear(latent_dim, latent_dim)),
            ])
        )

        self.patch_encoder = nn.Sequential(
            collections.OrderedDict([
                ("patchlin1", nn.Linear(patch_inp, latent_dim)),
                ("patchrelu1", nn.ReLU()),
                ("patchlin2", nn.Linear(latent_dim, latent_dim)),
            ])
        )

        self.positive_anchor_weight_mask = nn.Sequential(
            collections.OrderedDict([
                ("positive_anchorwei1", nn.Linear(latent_dim, num_ts * (positive_anchor_patch_inp))),
                ("positive_anchorwei1relu1", nn.ReLU()),
                ("positive_anchorwei2", nn.Linear(latent_dim, num_ts)),
                ("positive_anchorwei2soft2", nn.Softmax()),
            ])
        )

        self.patch_weight_mask = nn.Sequential(
            collections.OrderedDict([
                ("patchwei1", nn.Linear(latent_dim, num_ts * (positive_anchor_patch_inp))),
                ("patchwei1relu1", nn.ReLU()),
                ("patchwei2", nn.Linear(latent_dim, num_ts)),
                ("patchwei2soft2", nn.Softmax())
            ])
        )
        # self.decoder = nn.Sequential(
        #     collections.OrderedDict([
        #         ("declin1", nn.Linear(decoder_input, latent_dim)),
        #         ("decrelu1", nn.ReLU()),
        #         ("decop", nn.Linear(latent_dim, 1)),
        #     ])
        # )

    def forward(self, positive_anchor_data, patch_data, l):

        # bs, num_ts, positive_anchor_dim = positive_anchor_d.shape
        # _,_, patch_dim = patch_d.shape
        bs_d, num_ts_d, positive_anchor_d = positive_anchor_data.shape

        positive_anchor_d = []
        for batch in range(bs_d):
            if int(num_ts_d*l[batch])+self.window_size > num_ts_d:
                pos_begin = np.random.randint(0, num_ts_d - self.window_size)
                pos_end = pos_begin + self.window_size
                interval = [pos_begin, pos_end]
            else:
                interval = [int(num_ts_d*l[batch]), int(num_ts_d*l[batch])+self.window_size]

            # print('interval', interval[0], interval[1])
            positive_anchor_d.append(positive_anchor_data[batch, interval[0]:interval[1], :].unsqueeze(0))

        positive_anchor_d = torch.cat(positive_anchor_d, dim=0).detach()

        bs, num_ts, positive_anchor_dim = positive_anchor_d.shape
        positive_anchor_weight_d = positive_anchor_d.view(bs, num_ts * positive_anchor_dim)

        positive_anchor_d = positive_anchor_d.view(bs*num_ts, positive_anchor_dim)

        positive_anchor_emb = self.positive_anchor_encoder(positive_anchor_d)
        positive_anchor_weight = self.positive_anchor_weight_mask(positive_anchor_weight_d)

        bs_d, num_ts_d, patch_d = patch_data.shape

        patch_d = []
        for batch in range(bs_d):
            if int(num_ts_d * l[batch]) + self.window_size > num_ts_d:
                pos_begin = np.random.randint(0, num_ts_d - self.window_size)
                pos_end = pos_begin + self.window_size
                interval = [pos_begin, pos_end]
            else:
                interval = [int(num_ts_d * l[batch]), int(num_ts_d * l[batch]) + self.window_size]

            patch_d.append(patch_data[batch, interval[0]:interval[1], :].unsqueeze(0))

        patch_d = torch.cat(patch_d, dim=0).detach()

        bs, num_ts, patch_dim = patch_d.shape
        patch_weight_d = patch_d.view(bs, num_ts * patch_dim)

        patch_d = patch_d.view(bs * num_ts, patch_dim)

        patch_emb = self.patch_encoder(patch_d)
        patch_weight = self.patch_weight_mask(patch_weight_d)

        positive_anchor_emb = positive_anchor_emb.view(bs, num_ts, positive_anchor_emb.shape[-1])*positive_anchor_weight
        patch_emb = patch_emb.view(bs, num_ts, patch_emb.shape[-1])*patch_weight
        positive_anchor_emb = torch.mean(positive_anchor_emb, dim=1)
        patch_emb = torch.mean(patch_emb, dim=1)

        positive_anchor_patch_emb = torch.cat([positive_anchor_emb,patch_emb], axis  = -1)

        return positive_anchor_patch_emb

class CORE(nn.Module):
    '''
    Core network is a recurrent network which maintains a behavior state.
    '''
    def __init__(self):
        super(CORE, self).__init__()

        self.fc_h = nn.Linear(256, 256)
        self.fc_g = nn.Linear(256,256)

    def forward(self, h, g):
        return F.relu(self.fc_h(h) + self.fc_g(g)) # recurrent connection
        # return F.relu(self.fc_h(h))

class LOCATION(nn.Module):
    '''
    Location network learns policy for sensing locations.
    '''
    def __init__(self, std):
        super(LOCATION, self).__init__()

        self.std = std
        self.fc = nn.Linear(256,1)

    def forward(self, h):
        h = h.detach()
        l_mu = self.fc(h)               # compute mean of Gaussian
        pi = Normal(l_mu, self.std)     # create a Gaussian distribution
        l = pi.sample()                 # sample from the Gaussian 
        logpi = pi.log_prob(l)          # compute log probability of the sample
        l = torch.sigmoid(l)               # squeeze location to ensure sensing within the boundaries of an image
        return logpi, l.detach()                 # logpi, l: B*2

class ACTION(nn.Module):
    '''
    Action network learn policy for task specific actions.
    In case of classification actions are possible classes.
    This network will be trained with supervised loss in case of classification.
    '''
    def __init__(self):
        super(ACTION, self).__init__()

        self.hidden = nn.Linear(256,32)
        self.fc = nn.Linear(32,1)

    def forward(self, h):
        hid = self.hidden(h)
        return self.fc(hid)  # Do not apply softmax as loss function will take care of it

class MODEL(nn.Module):
    '''
    Model combines all the previous elements
    '''
    def __init__(self, window_size, std):
        super(MODEL, self).__init__()

        # self.glimps = GLIMPSE(im_sz, channel, glimps_width, scale)
        self.glimps = GLIMPSE(window_size)
        self.core   = CORE()
        self.location = LOCATION(std)
        self.action = ACTION()

    def initialize(self, B, device):
        self.state = torch.zeros(B,256).to(device)    # initialize states of the core network
        self.l = torch.rand((B,1)).to(device)   # start with a glimpse at random location

    def forward(self, positive_anchor_data, patch_data):
        # g = self.glimps(x,self.l)
        # glimpse encoding
        g = self.glimps(positive_anchor_data, patch_data, self.l)
        self.state = self.core(self.state, g)         # update state of a core network based on new glimpse
        logpi, self.l = self.location(self.state)     # predict location of next glimpse
        a = self.action(self.state)                   # predict task specific actions
        return logpi, a

class LOSS(nn.Module):
    '''
    Loss function is tailored for the reward received at the end of the episode.
    Location network is trained with REINFORCE objective.
    Action network is trained with supervised objective.
    '''
    def __init__(self, T, gamma, device):
        super(LOSS, self).__init__()

        self.baseline = nn.Parameter(0.1*torch.ones(1,1).to(device), requires_grad = True) # initialize baseline to a reasonable value
        self.T = T                                                                         # length of an episode
        self.gamma = gamma # discount factor
        self.device = device

    def initialize(self, B):
        self.t = 0
        self.logpi = []

    def compute_reward(self, predict, label):
        pred = 1*(torch.sigmoid(predict)>0.5)
        return (label.detach()==pred.detach()).squeeze().float() # reward is 1 if the classification is correct and zero otherwise


    def forward(self, predict, label, logpi):
        self.t += 1
        self.logpi += [logpi]
        if self.t==self.T:
            R = self.compute_reward(predict, label)                     # reward is given at the end of the episode
            criterion = nn.BCEWithLogitsLoss(reduction='sum').to(self.device)
            a_loss = criterion(predict, label)         # supervised objective for action network
            l_loss = 0
            R_b = (R - self.baseline.detach())                      # centered rewards
            for logpi in reversed(self.logpi):
                l_loss += - (logpi.sum(-1) * R_b).sum()             # REINFORCE
                R_b = self.gamma * R_b                              # discounted centered rewards (although discount factor is always 1)
            b_loss = ((self.baseline - R)**2).sum()                 # minimize SSE between reward and the baseline
            return a_loss , l_loss , b_loss, R.sum()
        else:
            return None, None, None, None
                
def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    '''
    Decay learning rate
    '''
    lr = lr * (decay_rate ** (epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



