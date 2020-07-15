import torch
import torch.nn as nn
from torch.autograd import Function


class BinaryLayer(nn.Module):
    def __init__(self):
        super().__init__()


# https://discuss.pytorch.org/t/custom-binarization-layer-with-straight-through-estimator-gives-error/4539/3
class BinaryFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


# https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/

if __name__ == "__main__":
    input = torch.randn(4, 4, requires_grad=True)
    model = BinaryFunction.apply
    output = model(input)
    loss = output.mean()
    loss.backward()

    print(input)
    print(input.grad)
    print(output)
