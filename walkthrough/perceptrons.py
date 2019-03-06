import sklearn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

torch.manual_seed(5)

n_pts = 100
centers = [
	[-0.5, 0.5],
	[0.5, -0.5]
]
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

x_data = torch.Tensor(X)
y_data = torch.Tensor(y).view(100, -1)

def scatter_plot():
	plt.scatter(X[y==0, 0], X[y==0, 1])
	plt.scatter(X[y==1, 0], X[y==1, 1])
	plt.show()


class Model(nn.Module):
	
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		pred = torch.sigmoid(self.linear(x)) #sigmoid activation function gives us continuous prob of class
		return pred

	def predict(self, x):
		pred = self.forward(x)
		if pred >= 0.5:
			return 1
		else:
			return 0


model = Model(2,1)

# print(list(model.parameters()))

def get_params():
	[w, b] = model.parameters()
	w1,w2 = w.view(2)
	b = b[0]
	return(w1.item(),w2.item(),b.item())


def plot_fit(title):
	plt.title = title
	# 0 = w1*x1 + w2*x2 + b // rewriting the eqn of the line y = mx + c to 0 = mx -y + c
	w1, w2, b  = get_params()
	x1 = np.array([-2., 2.])
	x2 = (w1*x1 + b)/-w2

	plt.plot(x1, x2, 'r')
	scatter_plot()
	plt.show()
	
# plot_fit('Initial Model')


criterion = nn.BCELoss() #binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
losses = []

for i in range(epochs):
	pred = model.forward(x_data)
	loss = criterion(pred, y_data)
	losses.append(loss)

	print('epoch:', i, 'loss:', loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')

# plt.show()

# plot_fit('trained model')
# plt.show()


point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])
point3 = torch.Tensor([1.0, 1.0])

plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'ro')
plt.plot(point3.numpy()[0], point3.numpy()[1], 'ko')

print("red point 1 prob = ", model.forward(point1).item())
print("red point 2 prob = ", model.forward(point2).item())
print("black point prob = ", model.forward(point3).item())

plot_fit('trained_model')

# ****************************************
# Cross-entropy definition and derivation
# ****************************************

# A continuous error function that gives a larger penalty to misclassified points that are further away from the decision surface
# The probability of seeing the data as it is, given the model we have, is the *product* of all the probabilities of each individual point. We want the 'maximum likelihood model', so we change the model until we reach this model.

# Products are not good to work with when we have many numbers. So instead we use sums. We use (the natural) log to turn products into sums. This is because log has a nice identity that log(a*b) is equal to log(a) + log(b). So instead of multiplying, we sum the (natural) logs. We use natural log just for convention, it would still work with log base 10. 

# Now, ln of a number between 0 and 1 is always negative, so we will neg it so we can work with positive numbers. 

#CROSS-ENTROPY is the sum of the negative logarithms of the probabilities. Good models have a low cross-entropy. 

# Correctly classified points with high prob have a very low negative log. Correctly classified points with lower prob have a slightly higher negative log. Incorrectly classified points have even higher negative logs, depending on the level of probability. So we can think of negative log of prob as the error. 


# cross-entropy = -sum(
#	 y*ln(p) 			// if y == 1, we just take ln(prob_of_being_a_one) and the other term is zero
#	 + (1-y)*ln(1-p)	// if y == 0, we take this term, which it ln(prob_of_being_a_zero) and the first term is nothing
# )

# where p is the probability of being y == 1

# So, cross-entropy is a measure of the error that we can use gradient descent on





