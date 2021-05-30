import torch as t

#Initializing tensor
import torch.cuda

#setting where the computations are performed gpy(cuda) vs cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = t.tensor([[1,2,3],[4,5,6]], dtype=t.float32,
                     device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#Other common initialization methods

x = t.empty(size = (3, 3))
print(x)

x = t.zeros((3, 3))
print(x)

x = t.rand((3, 3))
print(x)

x = t.ones((3, 3))
print(x)

#Identity matrix I
x = t.eye(3, 3)
print(x)

x = t.arange(start=0, end=5, step=1)
print(x)

x = t.linspace(start=0.1, end=1, steps=10)
print(x)

x = t.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)

x = t.empty(size=(1,5)).uniform_(0, 1)
print(x)

x = t.diag(torch.ones(3))
print(x)


# How to initialize and convert tensors to other types (int, float, double)
tensor = t.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

# Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = t.from_numpy(np_array)
np_array_back = tensor.numpy()



#===========================================================================
# Tensor math & comparison operations
#===========================================================================

x = t.tensor([1,2,3])
y = t.tensor([9,8,7])

# addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x,y)
z3 = x + y

# subtraction
s = x -y

#division
d = torch.true_divide(y, 2)

# inplace operations
i = t.zeros(3)
i.add_(x)

# exponentiation
z = x.pow(2)
z = x ** 2


# simple comparison
z = x > 0
z = x < 0

# matrix multiplication
x1 = t.rand((2, 5))
x2 = t.rand((5, 3))

x3 = t.mm(x1, x2)
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = t.rand((5, 5))
matrix_exp.matrix_power(3)

# element wise mult
z = x * y

# dot product
z = t.dot(x, y)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = t.rand(batch, n, m)
tensor2 = t.rand(batch, m, p)

out_bmm = t.bmm(tensor1, tensor2)

# example of broadcasting (automatically expanding vars to math the operation)
x1 = t.rand((5,5))
x2 = t.rand((1,5))

z = x1 - x2
z = x1 ** x2


# other useful tensor operations
sum_x = t.sum(x, dim=0)
values, indices = t.max(x, dim=0)
values, indices = t.min(x, dim=0)
abs_x = t.abs(x)
# returning the indcex of the max/min value
z = t.argmax(x, dim=0)
z = t.argmin(x, dim=0)
mean_x = t.mean(x.float(), dim=0)
# checks for equal values
z = t.eq(x, y)
sorted_y, indices = t.sort(y, dim=0, descending=False)
# restricts the values of x to the given range [min,max] values outside of this range are set to either min (if val smaller than ragen) or max
z = t.clamp(x, min=0, max=10)

x = t.tensor([1,0,1,1,1], dtype=t.bool)
# any vall of x true?
z = t.any(x)
# all val of x true?
z = t.all(x)



#===========================================================================
# Tensor  indexing
#===========================================================================


batch_size = 10
features = 25
x = t.rand((batch_size, features))

print(x[0].shape)
print(x[:, 0].shape)

print(x[2, 0:10])

x[0, 0] = 100

#fancy indexing
x = t.arange(10)
x.add_(1)
indices = [2, 5, 8]
print(x[indices])

x = t.rand((3, 5))
rows = t.tensor([1,0])
cols = t.tensor([4, 0])
print(x[rows, cols]) # not sure whats happening here?

# mroe advanced indexing
x = t.arange(10)
print(x)
# indexing depending on condition
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# useful operations
print(t.where(x > 5, x, x*2))
print(t.tensor([0, 0, 1, 2, 3, 4]).unique())
print(tensor1.ndimension())
print(tensor1.numel())


#===========================================================================
# Tensor  reshaping
#===========================================================================

x = t.arange(9)

x_3x3 = x.view(3, 3) # faster but more complex to use
x_3x3 = x.reshape(3, 3)

y = x_3x3.t()
print(y.contiguous().view(9))

x1 = t.rand((2, 5))
x2 = t.rand((2, 5))
print(t.cat((x1, x2), dim=0).shape)
print(t.cat((x1, x2), dim=1).shape)

# flatten the matrix
z = x1.view(-1)
print(z.shape)

batch = 64
x = t.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)
print(z.shape)

x = t.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(0))
print(x.unsqueeze(1))


x = t.arange(10).unsqueeze(0).unsqueeze(1)
z = x.squeeze(1)
print(z.shape)






exit(0)

