import torch
import torchvision
from torch import nn

# An instance of your model.
model = torchvision.models.resnet18().to('cuda')

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).to('cuda')

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1, 3, 224, 224).to('cuda'))
print(output[0, :5])

traced_script_module.save("traced_resnet_model_gpu.pt")