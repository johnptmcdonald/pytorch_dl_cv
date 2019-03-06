import torch
import torch.nn as nn

# y = mx + c

x = torch.randn(50, 1)*10
y = x + torch.randn(50,1)*3

torch.manual_seed(1)

m = torch.tensor(3., requires_grad=True)
c = torch.tensor(2., requires_grad=True)

def forward(x):
	pred = m*x + c
	return pred


print(forward(x))

criterion = nn.MSELoss()
optimizer = torch.optim.SGD([m,c], lr=0.001)

epochs = 50

losses = []

for i in range(epochs):
	y_pred = forward(x)
	loss = criterion(y_pred, y)
	print('epoch:', i, 'loss:', loss.item())
	optimizer.zero_grad()
	loss.backward() # computes the gradients for all tensors used in the loss calculation that have requires_grad=True
	optimizer.step()

