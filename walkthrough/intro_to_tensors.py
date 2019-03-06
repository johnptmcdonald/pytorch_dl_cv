import torch
import numpy as np
import matplotlib.pyplot as plt

# *****************************
# Tensors 1
# - torch.view
# - torch.from_numpy()
# *****************************

print('*****')
print('tensors 1')
v = torch.tensor([1,2,3,4,5,6])

print(v.shape)
v = v.view(2,-1)

print(v)

a = np.array([1,2,3,4,5])

converted_a = torch.from_numpy(a)
print(converted_a)


# *****************************
# Tensors 2
# - elementwise addition and multiplication
# - torch.dot(tensor, tensor)
# - torch.linspace(start, stop, howmany)
# - torch.exp(tensor)
# - torch.numpy(tensor)
# *****************************

print('*****')
print('tensors 2')
t_one = torch.tensor([1,2,3])
t_two = torch.tensor([1,2,3])

dot_product = torch.dot(t_one,t_two)

x = torch.linspace(0,10,50)
y = torch.exp(x)

# plt.plot(x.numpy(), y.numpy())
# plt.show()

# *****************************
# Tensors 3
# - rank 3 tensor slicing
# *****************************
print('*****')
print('tensors 3')

one_d = torch.arange(0, 9)
print(one_d)
two_d = one_d.view(3,3)
print(two_d)

x = torch.arange(18).view(3,2,3)
print(x)
print(x[1, 0:, 0:])

# *****************************
# Tensors 4
# - matrix multiplication with matmul or mm (not dot)
# *****************************
print('*****')
print('tensors 4')

mat_a = torch.tensor([0,3,5,5,5,2]).view(2,3)
mat_b = torch.tensor([3,4,3,-2,4,-2]).view(3,2)

print(mat_a, mat_b)
print(torch.mm(mat_a, mat_b))
print(mat_a @ mat_b)


# *****************************
# Tensors 5
# - derivatives
# *****************************
print('*****')
print('tensors 5')

x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 6*x + 1
y.backward()

print(x.grad)

x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3

y.backward()
print(x.grad)
print(z.grad)






