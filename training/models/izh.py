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
        return grad * (1.0 / (1.0 + torch.abs(input)) ** 2)


class IzhikevichNode(nn.Module):
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, threshold=30.0, dt=1.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.threshold = threshold
        self.dt = dt
        self.v = None
        self.u = None

    def forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, self.c)
            self.u = self.b * self.v

        v_inc = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x) * self.dt
        self.v = self.v + v_inc
        u_inc = self.a * (self.b * self.v - self.u) * self.dt
        self.u = self.u + u_inc

        spike = SpikeFunction.apply(self.v - self.threshold)

        self.v = self.v * (1 - spike) + self.c * spike
        self.u = self.u + self.d * spike

        return spike

    def reset(self):
        self.v = None
        self.u = None
