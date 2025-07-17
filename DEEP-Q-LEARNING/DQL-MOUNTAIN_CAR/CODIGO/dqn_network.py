import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self,states_dim,actions_dim):
        super().__init__()
        self.l1=nn.Linear(in_features=states_dim,out_features=128)  #64
        self.r1=nn.ReLU()

        self.l2=nn.Linear(in_features=128,out_features=128)
        self.r2=nn.ReLU()

        self.l3=nn.Linear(in_features=128,out_features=actions_dim)

    def forward(self,x):
        x=self.l1(x)
        x=self.r1(x)
        
        x=self.l2(x)
        x=self.r2(x)
        
        x=self.l3(x)
        
        return x
