import torch
import torch.nn as nn
import pytorch_lightning as pl

class RecognizeNet(nn.Module):
    r""" Calculate the posterior probability p (z | x, y) of the network;The x, y, as the last step the output of the decoder """
    def __init__(self, x_size,  # post dimension
                 y_size,  # response dimension
                 latent_size,  # 
                 dims):  # 
        super(RecognizeNet, self).__init__()
        assert len(dims) >= 1  # At least two layer perceptron

        dims = [x_size+y_size] + dims + [latent_size*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.latent_size = latent_size
        self.mlp = nn.Sequential()
        for idx, (x, y) in enumerate(zip(dims_input[:-1], dims_output[:-1])):
            self.mlp.add_module(f'linear{idx}', nn.Linear(x, y))  # Linear layer
            self.mlp.add_module(f'activate{idx}', nn.Tanh())  # The activation layer
        self.mlp.add_module('output', nn.Linear(dims_input[-1], dims_output[-1]))

    def forward(self, x,  # [batch, x_size]
                y):  # [batch, y_size]
        x = torch.cat([x, y], 1)  # [batch, x_size+y_size]
        self.mlp.to(x.device)
        predict = self.mlp(x)  # [batch, la+tent_size*2]
        mu, logvar = predict.split([self.latent_size]*2, 1)
        return mu, logvar
