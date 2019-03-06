import sklearn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

n_pts = 100
centers = [
	[-0.5, 0.5],
	[0.5, -0.5]
]
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

x_data = torch.tensor(X)
y_data = torch.tensor(y)

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

plt.show()


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





