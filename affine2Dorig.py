"""
This is the module for the RealNVP-style affine layer, for a 2D input variable.
"""

#import nln

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from copy import deepcopy
import math
import numpy as np


class affine2(nn.Module):
    def __init__(self, n, hidden=128, bias=True):
        super(affine2, self).__init__()

        self.n = n
        self.d = nn.Dropout(0.)
#        self.b = nn.BatchNorm1d(2)
        #Initialize everything.
#        y = self.b(torch.randn(100, 2))
#        self.b.weight.data.fill_(1.)
#        self.b.bias.data.fill_(0.)

        self.m_up1   = nn.Linear(1, hidden)
        self.m_fc1a  = nn.Linear(hidden, hidden)
        self.m_fc1b  = nn.Linear(hidden, hidden)
        self.m_down1 = nn.Linear(hidden, 1)
        
        self.m_up2   = nn.Linear(1, hidden)
        self.m_fc2a  = nn.Linear(hidden, hidden)
        self.m_fc2b  = nn.Linear(hidden, hidden)
        self.m_down2 = nn.Linear(hidden, 1)

        self.b_up1   = nn.Linear(1, hidden)
        self.b_fc1a  = nn.Linear(hidden, hidden)
        self.b_fc1b  = nn.Linear(hidden, hidden)
        self.b_down1 = nn.Linear(hidden, 1)
        
        self.b_up2   = nn.Linear(1, hidden)
        self.b_fc2a  = nn.Linear(hidden, hidden)
        self.b_fc2b  = nn.Linear(hidden, hidden)
        self.b_down2 = nn.Linear(hidden, 1)
        
        #Fight initial nan's
        self.m_down1.weight.data /= math.sqrt(hidden)
        self.b_down1.weight.data /= math.sqrt(hidden)
        self.m_down2.weight.data /= math.sqrt(hidden)
        self.b_down2.weight.data /= math.sqrt(hidden)
        self.m_down1.bias.data /= math.sqrt(hidden)
        self.b_down1.bias.data /= math.sqrt(hidden)
        self.m_down2.bias.data /= math.sqrt(hidden)
        self.b_down2.bias.data /= math.sqrt(hidden)

      
    def forward(self, x):
#        y  = x + 0
        s  = x.size(0)

        h1m = self.d(F.relu(self.m_up1(x[:, 0].view(s, 1))))
        h1m = self.d(F.relu(self.m_fc1a(h1m)))
        h1m = self.d(F.relu(self.m_fc1b(h1m)))
        m1  = self.m_down1(h1m).view(s)        

        h1b = self.d(F.relu(self.b_up1(x[:, 0].view(s, 1))))
        h1b = self.d(F.relu(self.b_fc1a(h1b)))
        h1b = self.d(F.relu(self.b_fc1b(h1b)))
        b1  = self.b_down1(h1b).view(s)
        
#        print(m1.size())
#        print(y[:, 1].size()) 
        y1 = torch.exp(m1)*x[:, 1] + b1

        h2m = self.d(F.relu(self.m_up2(y1.view(s, 1))))
        h2m = self.d(F.relu(self.m_fc2a(h2m)))
        h2m = self.d(F.relu(self.m_fc2b(h2m)))
        m2  = self.m_down2(h2m).view(s)        

        h2b = self.d(F.relu(self.b_up2(y1.view(s, 1))))
        h2b = self.d(F.relu(self.b_fc2a(h2b)))
        h2b = self.d(F.relu(self.b_fc2b(h2b)))
        b2  = self.b_down2(h2b).view(s)

        y0 = torch.exp(m2)*x[:, 0] + b2
       
#       Silly as it is the "+ 0" rewrites the tensor in another place, contiguosly.
#        y = torch.stack((y0, y1)).t() + 0
        y = (torch.stack((y0, y1)).t() + 0)

        return y, m1 + m2 
    
    def extra_repr(self):
        return 'n={}'.format( \
            self.n \
        )

    def pushback(self, y):
#        x = y + 0
        s = y.size(0)
#        self.eval()
#        z = ((y - self.b.bias)/self.b.weight)*torch.sqrt(self.b.eps + self.b.running_var) + self.b.running_mean

#        h2m = F.relu(self.m_up2(z[:, 1].view(s, 1)))
        h2m = F.relu(self.m_up2(y[:, 1].view(s, 1)))
        h2m = F.relu(self.m_fc2a(h2m))
        h2m = F.relu(self.m_fc2b(h2m))
        m2  = self.m_down2(h2m).view(s)        

#        h2b = F.relu(self.b_up2(z[:, 1].view(s, 1)))
        h2b = F.relu(self.b_up2(y[:, 1].view(s, 1)))
        h2b = F.relu(self.b_fc2a(h2b))
        h2b = F.relu(self.b_fc2b(h2b))
        b2  = self.b_down2(h2b).view(s)

#        x0 = (z[:, 0] - b2)/torch.exp(m2)
        x0 = (y[:, 0] - b2)/torch.exp(m2)

        h1m = F.relu(self.m_up1(x0.view(s, 1)))
        h1m = F.relu(self.m_fc1a(h1m))
        h1m = F.relu(self.m_fc1b(h1m))
        m1  = self.m_down1(h1m).view(s)
        
        h1b = F.relu(self.b_up1(x0.view(s, 1)))
        h1b = F.relu(self.b_fc1a(h1b))
        h1b = F.relu(self.b_fc1b(h1b))
        b1  = self.b_down1(h1b).view(s)
        
        x1 = (y[:, 1] - b1)/torch.exp(m1)

        return torch.stack((x0, x1)).t(), m1 + m2 


    
