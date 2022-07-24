"""
How to use Torrchscript to convert Modules
"""

import torch  # This is all you need to use both PyTorch and TorchScript!
from torch import nn


print(torch.__version__)

class MyDecisionGate(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h

# Initialize cell with other module as argument
my_cell = MyCell(MyDecisionGate())
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
print("Original:", my_cell(x, h))
print("Traced:", traced_cell(x, h))

# Torchscript IR (graph) Python-syntax interpretation of the code
print(traced_cell.graph)

print(traced_cell.dg.code)
print(traced_cell.code)

# !!! For some inputs, the outputs will be different. Why? because the tracing of dg is not correct.
# !!! It was only fit for the type of input initially provided
# !!! The if-else branch is not seen in the .code output of dg.
# !!! This is because tracing just runs the input and captures the flow.
# !!! This is not generalizable
# !!! For this, script compiler must be used