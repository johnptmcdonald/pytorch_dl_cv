import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# we want to model this linear equation:
# y = wx + b 

# ******************************
# Manual Linear model
# ******************************
# We can do it manually:
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

def forward(x):
	y = w*x + b
	return y


# ******************************
# Pytorch's Linear model
# ******************************

# or we can use pytorch's Linear model
torch.manual_seed(1) # random seed

model = nn.Linear(in_features=1, out_features=1)
print(model.bias, model.weight) # bias and weigth are random

x = torch.tensor([[2.], [3.], [4.]])
print(model(x)) # get predictions


# ******************************
# Custom models
# ******************************

class LR(nn.Module):
	
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		pred = self.linear(x)
		return pred


model = LR(1,1)
print(list(model.parameters()))
print(model.forward(x))



