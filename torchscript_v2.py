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

# Script cell - makes graph on how order of operations is applied including control flows
scripted_gate = torch.jit.script(MyDecisionGate())
my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

# Torchscript IR (graph) Python-syntax interpretation of the code - Control flows visible now
print(scripted_gate.code)
print(scripted_cell.code)

# Inputs changed - Test new scripted module
print("Inputs changed")
x, h = torch.rand(3, 4), torch.rand(3, 4) # torch.rand(5, 4), torch.rand(5, 4)  this also works

print("Original:", my_cell(x, h))
print("Scripted:", scripted_cell(x, h))