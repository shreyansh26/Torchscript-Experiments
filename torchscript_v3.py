"""
Some situations call for using tracing rather than scripting (e.g. a module has many architectural decisions that are made based on constant Python values that we would like to not appear in TorchScript). In this case, scripting can be composed with tracing: torch.jit.script will inline the code for a traced module, and tracing will inline the code for a scripted module.

An example of the first case:
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
        return new_h, new_h
        
scripted_gate = torch.jit.script(MyDecisionGate())
x, h = torch.rand(3, 4), torch.rand(3, 4)

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)

"""
And an example of the second case:
"""

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)