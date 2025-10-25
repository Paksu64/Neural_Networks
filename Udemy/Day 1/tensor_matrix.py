import torch 

# all rows must have an equal number of columns
X = torch.tensor([
    [10, 49],
    [38, 9],
    [100, 78],
    [150, 94]
])

print(X)

#4 rows, 1 column
print(X.shape)
print(X.size())

#4 rows (dimension 0)
print(X.size(0))

#1 column (dimension 1)
print(X.size(1))

#3rd row 1st column
print(X[2, 0].item())

#All elements on row 2
print(X[1, :])