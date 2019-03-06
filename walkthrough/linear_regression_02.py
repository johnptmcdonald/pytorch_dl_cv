import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# we want to model this linear equation:
# y = wx + b 

x = torch.randn(100,1)*10 #100 rows, 1 column
y = x + 3*torch.randn(100,1)

plt.plot(x.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

class LR(nn.Module):
	
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		pred = self.linear(x)
		return pred


torch.manual_seed(1)
model = LR(1,1)

[w, b] = model.parameters()

def get_params():
	return (w[0][0].item(), b[0].item()) # .item() returns the scalar value

def plot_fit(title):
	plt.title = title
	w1, b1 = get_params()
	x1 = np.array([-30,30])
	y1 = w1*x1 + b1
	plt.plot(x1, y1, 'r')
	plt.scatter(x, y)
	plt.show()

plot_fit('initial model')

