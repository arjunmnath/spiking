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


class HHNode(nn.Module):
    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387, threshold=-20.0, dt=0.01):
        super().__init__()
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.threshold = threshold
        self.dt = dt

        self.v = None
        self.m = None
        self.h = None
        self.n = None

    def alpha_m(self, V): return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))
    def beta_m(self, V):  return 4.0 * torch.exp(-(V + 65.0) / 18.0)
    def alpha_h(self, V): return 0.07 * torch.exp(-(V + 65.0) / 20.0)
    def beta_h(self, V):  return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
    def alpha_n(self, V): return 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))
    def beta_n(self, V):  return 0.125 * torch.exp(-(V + 65) / 80.0)

    def forward(self, I):
        if self.v is None or self.v.shape != I.shape:
            self.v = torch.full_like(I, -65.0)
            self.m = self.alpha_m(self.v) / (self.alpha_m(self.v) + self.beta_m(self.v))
            self.h = self.alpha_h(self.v) / (self.alpha_h(self.v) + self.beta_h(self.v))
            self.n = self.alpha_n(self.v) / (self.alpha_n(self.v) + self.beta_n(self.v))

        I_Na = self.g_Na * self.m ** 3 * self.h * (self.v - self.E_Na)
        I_K = self.g_K * self.n ** 4 * (self.v - self.E_K)
        I_L = self.g_L * (self.v - self.E_L)

        dV = (I - I_Na - I_K - I_L) / self.C_m
        self.v = self.v + dV * self.dt

        dm = self.alpha_m(self.v) * (1 - self.m) - self.beta_m(self.v) * self.m
        dh = self.alpha_h(self.v) * (1 - self.h) - self.beta_h(self.v) * self.h
        dn = self.alpha_n(self.v) * (1 - self.n) - self.beta_n(self.v) * self.n

        self.m = self.m + dm * self.dt
        self.h = self.h + dh * self.dt
        self.n = self.n + dn * self.dt

        spike = SpikeFunction.apply(self.v - self.threshold)
        return spike

    def reset(self):
        self.v = None
        self.m = None
        self.h = None
        self.n = None
