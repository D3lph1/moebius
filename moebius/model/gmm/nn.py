import torch
from torch import nn

class ParameterGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, softmaxable: int):
        super(ParameterGenerator, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.layer3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.layer4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.layer5 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.activation = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        self.softmaxable = softmaxable

    def forward(self, x):
        y = self.layer1(x)
        y = self.activation(y)

        y = self.layer2(y)
        y = self.activation(y)

        y = self.layer3(y)
        y = self.activation(y)

        y = self.layer4(y)
        y = self.activation(y)

        y = self.layer5(y)

        first_slice = y[:, 0:self.softmaxable]
        second_slice = y[:, self.softmaxable:]
        tuple_of_activated_parts = (
            self.softmax(first_slice),
            second_slice
        )

        y = torch.cat(tuple_of_activated_parts, dim=1)

        return y

    @staticmethod
    def load(pathname: str):
        checkpoint = torch.load(pathname)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()

        return model

    def save(self, pathname: str):
        checkpoint = {
            'model': self,
            'state_dict': self.state_dict()
        }

        torch.save(checkpoint, pathname)
