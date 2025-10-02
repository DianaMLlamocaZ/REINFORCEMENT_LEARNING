import torch

class Modelo(torch.nn.Module):
    def __init__(self,st_space,act_space):
        super().__init__()
        self.l1=torch.nn.Linear(in_features=st_space,out_features=32) #64
        self.r1=torch.nn.ReLU()

        self.l2=torch.nn.Linear(in_features=32,out_features=64)
        self.r2=torch.nn.ReLU()

        self.l3=torch.nn.Linear(in_features=64,out_features=act_space)
        self.sm=torch.nn.Softmax(dim=1)
        

    def forward(self,x):
        x=self.l1(x)
        x=self.r1(x)

        x=self.l2(x)
        x=self.r2(x)

        x=self.l3(x)
        x=self.sm(x)

        return x #Devuelvo las probabilidades