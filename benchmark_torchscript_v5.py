import torch
import numpy as np

model = torch.jit.load('traced_resnet_model.pt')

example = torch.rand(1, 3, 224, 224)

output = model(example)

print(output.shape)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 1000
timings = np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(example)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(example)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)

print("Mean time: {:.2f}ms".format(mean_syn))
print("Total time: {:.2f}ms".format(np.sum(timings)))