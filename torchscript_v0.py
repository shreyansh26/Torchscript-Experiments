"""
How to use Torchscript
"""

import torch  # This is all you need to use both PyTorch and TorchScript!
from torch import nn


print(torch.__version__)

class MyCell(nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

# Initialize cell
my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)

# Current cell
print(my_cell)
print(my_cell(x, h))

# Trace cell - makes graph on how order of operations is applied
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
print(traced_cell(x, h))

# Inputs changed and run on original and traced cell
print("Inputs changed")
x, h = torch.rand(3, 4), torch.rand(3, 4) # torch.rand(5, 4), torch.rand(5, 4)  this also works
print(my_cell(x, h))
print(traced_cell(x, h))

# Torchscript IR (graph) and Python-syntax interpretation of the code
print(traced_cell.graph)
print(traced_cell.code)