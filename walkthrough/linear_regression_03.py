import torch
import torch.nn as nn

torch.manual_seed(1)

x = torch.randn(100,1)*10 #100 rows, 1 column
y = x + 3*torch.randn(100,1)


class LR(nn.Module):
	
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		pred = self.linear(x)
		return pred

model = LR(1,1)


# ******************************
# Gradient descent
# ******************************

criterion = nn.MSELoss() #define the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #define the optimizer (stochastic gradient descent), and the thing we want to optimize (the model parameters)
epochs = 100

losses = []

for i in range(epochs):
	y_pred = model.forward(x)
	loss = criterion(y_pred, y)
	print('epoch:', i, 'loss:', loss.item())
	losses.append(loss)

	optimizer.zero_grad() #the optimizer zeroes out the grads of any tensors it is optimizing
	loss.backward() #the model parameters (which are tensors) store their gradient
	optimizer.step() #the optimizer goes through each tensor it should be optimizing, and steps each one in the -ve direction of its stored gradient



