import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Create tensors

# Scalar tensors
scalar = torch.tensor(7)
print(scalar)
print(f"The dimension of the scalar tensor is:  {scalar.ndim}")

# Get tensor back as Python int
print(scalar.item())

# Vector tensor
vector = torch.tensor([7,7])
print(vector)
print(f"The dimension of the vector tensor is:  {vector.ndim}") # dimensions could be defined by the brackets
print(vector.shape)

# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX.ndim)
print(f"First dimension of the matrix: {MATRIX[0]}")
print(MATRIX.shape)

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
print(TENSOR.shape) # third braket has 3 elements, second has another three elements of three, first has one element of three of three


## Random tensors

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)
random_tensor2 = torch.rand(2, 2 ,2)
print(random_tensor2)
print(random_tensor2.ndim)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size = (3, 224, 224)) # height, width, colour channels (R, G, B)
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)
print(random_image_size_tensor)