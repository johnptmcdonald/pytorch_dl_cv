import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets

n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

x_data = torch.Tensor(X)
y_data = torch.Tensor(y).view(n_pts, -1)

def scatter_plot():
	plt.scatter(X[y==0, 0], X[y==0, 1])
	plt.scatter(X[y==1, 0], X[y==1, 1])
	plt.show()

class Model(nn.Module):
	
	def __init__(self, input_size, H1, output_size):
		super().__init__()
		self.linear1 = nn.Linear(input_size, H1)
		self.linear2 = nn.Linear(H1, output_size)

	def forward(self, x):
		x = torch.sigmoid(self.linear1(x)) #sigmoid activation function gives us continuous prob of class
		x = torch.sigmoid(self.linear2(x))
		return x

	def predict(self, x):
		pred = self.forward(x)
		if pred >= 0.5:
			return 1
		else:
			return 0

torch.manual_seed(2)
model = Model(2,4,1)
print(list(model.parameters()))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.2) #Adam has a dynamic learning rate, highly recommended

epochs = 1000
losses = []

for i in range(epochs):
	pred = model.forward(x_data)
	loss = criterion(pred, y_data)
	losses.append(loss)
	print('epoch:', i, ', loss:', loss)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


plt.plot(losses)
plt.show()

def plot_decision_boundary(X, y):
	"""
	Plots every single point in the entire space of x and y
	"""
	x_span = np.linspace(min(X[:,0])-0.25, max(X[:,0])+0.25)
	y_span = np.linspace(min(X[:,1])-0.25, max(X[:,1])+0.25)
	xx, yy = np.meshgrid(x_span, y_span)
	
	grid = torch.Tensor(np.c_[xx.ravel(),yy.ravel()])
	print(grid.shape)
	pred_func = model.forward(grid)

	z = pred_func.view(xx.shape).detach().numpy()
	plt.contourf(xx,yy,z)
	


plot_decision_boundary(X, y)
scatter_plot()
plt.show()








