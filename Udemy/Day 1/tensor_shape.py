import torch

b = torch.tensor(32.0)
w1 = torch.tensor(1.9)

X1 = torch.tensor([1.0, 2.0, 3.0, 4.0])

y = 1*b + w1*X1

#first element of y
print(y[0].item())

#none because tensor is scalar
print(b.shape)

#4 values because X1 is a vector containing 4 values
print(X1.shape)