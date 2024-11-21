import torch
import torchvision

norm = torchvision.transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
inp = torch.rand(1, 3, 256, 256)


@torch.compile(backend="eager", fullgraph=True)
def fn(x):
    return norm(x)


fn(inp)
