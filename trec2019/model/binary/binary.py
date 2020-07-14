import torch
import torch.nn as nn
from torch.autograd import Function


# https://discuss.pytorch.org/t/custom-binarization-layer-with-straight-through-estimator-gives-error/4539/3
class BinaryLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.register_buffer("input", input)
        res = torch.sign(input)
        res.register_hook(self._backward_hook)
        return res

    def _backward_hook(self, grad):
        # get saved input
        for name, buf in self.named_buffers():
            if name in ["input"]:
                input = buf

        # clip
        grad[input > 1] = 0
        grad[input < -1] = 0
        return grad


# https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = lambda x: x.view(-1)
        self.fc1 = nn.Linear(160, 5)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x.register_hook(
            lambda grad: torch.clamp(grad, min=0)
        )  # No gradient shall be backpropagated
        # conv outside less than 0

        # print whether there is any negative grad
        x.register_hook(
            lambda grad: print("Gradients less than zero:", bool((grad < 0).any()))
        )
        return self.fc1(self.flatten(x))


if __name__ == "__main__":
    # input = torch.randn(4, 4, requires_grad=True) * 10
    # model = BinaryLayer()
    # output = model(input)
    # loss = output.mean()
    # loss.backward()

    # print(input)
    # print(input.grad)

    net = myNet()

    for name, param in net.named_parameters():
        # if the param is from a linear and is a bias
        if "fc" in name and "bias" in name:
            param.register_hook(lambda grad: torch.zeros(grad.shape))

    out = net(torch.randn(1, 3, 8, 8))

    (1 - out).mean().backward()

    print("The biases are", net.fc1.bias.grad)  # bias grads are zero
