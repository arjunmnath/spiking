import torch
import torch.nn as nn

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = grad_output.clone()
        return grad * (1.0 / (1.0 + torch.abs(input))**2)


class LIFNode(nn.Module):
    def __init__(self, tau=0.5, threshold=1.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.v = None

    def forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)

        self.v = self.tau * self.v + x
        spike = SpikeFunction.apply(self.v - self.threshold)
        self.v = self.v * (1 - spike)

        return spike

    def reset(self):
        self.v = None