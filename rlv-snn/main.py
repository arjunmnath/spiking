"""
RLVSL-SNN: Retina-LGN-V1 Structure-like Spiking Neuron Network
================================================================
Intel Lava SDK Implementation

Reference: Ji et al., "A Retina-LGN-V1 Structure-like Spiking Neuron Network
           for Image Feature Extraction", ICVIP 2021
DOI: https://doi.org/10.1145/3511176.3511197

Architecture:
  • Retina Sub-Network  : Photoreceptor → Bipolar Cell → Ganglion Cell
  • LGN Sub-Network     : Input Layer → Output Layer  (with V1-L6 feedback)
  • V1  Sub-Network     : L6 → L4 (Gabor/color) → L2/3
  • FC  SNN             : PyTorch ANN trained with backprop, then converted to SNN

Learning:
  • RLVSL-SNN : additive STDP, nearest-neighbour scheme
  • FC SNN    : ANN-to-SNN weight conversion (back-propagation on ANN proxy)

Data:
  • PyTorch MNIST data loaders + stochastic spike encoder
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
# ==============================================================================
# SECTION 1 — IMPORTS
# ==============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")
torch.set_flush_denormal(True)

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
# ── Lava core ──────────────────────────────────────────────────────────────────
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF


# ==============================================================================
# SECTION 2 — CONFIGURATION
# ==============================================================================
class Config:
    """All hyper-parameters from the paper (Section 3.2.1 and Table 1-3)."""

    # ── image ──────────────────────────────────────────────────────────────────
    IMG_H: int = 28
    IMG_W: int = 28

    # ── time ───────────────────────────────────────────────────────────────────
    DT: float = 0.1          # time step  (ms)
    T_WINDOW: int = 40        # time steps per image (paper Table 2)

    # ── LIF neurons : BCLon / BCMon / BCSon / RBC ──────────────────────────────
    TAU_BC: float = 1.0       # τ  (ms)
    RM_BC: float = 10.0       # Rm (Ω)
    VREST_BC: float = 1.0     # Vrest (V)
    VTH: float = 1.0          # threshold potential

    # ── LIF neurons : all other layers ─────────────────────────────────────────
    TAU_LIF: float = 0.5
    RM_LIF: float = 5.0
    VREST_LIF: float = 0.0

    # ── Graded neurons : photoreceptors ────────────────────────────────────────
    TAU_GRADE: float = 0.1
    RM_GRADE: float = 1.0
    VREST_GRADE: float = 1.0
    EP: float = 0.5           # stopping potential Ep
    VSL: float = 0.2          # slope control Vsl

    # ── Receptive-field radii ───────────────────────────────────────────────────
    RF_RBC: int = 3           # Rod → RBC Gaussian RF
    RF_GC_INH: int = 1        # surround inhibition in GC layer
    RF_LGN_INH: int = 1       # inhibition from GCoff to LGN input
    RF_V1: int = 1            # V1 L4 → L2/3 pooling radius

    # ── V1 channels ────────────────────────────────────────────────────────────
    N_FORM: int = 8           # V1-L4 form-pathway channels (optimal, Table 3)
    N_COLOR: int = 3          # V1-L4 colour-pathway channels
    GABOR_RF: int = 7         # Gabor kernel size (7×7)

    # ── STDP (additive, nearest-neighbour) ─────────────────────────────────────
    TAU_PLUS: float = 20.0    # τ+ (ms)
    TAU_MINUS: float = 20.0   # τ− (ms)
    A_PRE: float = 0.0001     # Apre
    A_POST: float = 0.000105  # Apost (paper: slightly asymmetric)
    W_MIN: float = 0.0
    W_MAX: float = 1.0

    # ── FC SNN training ────────────────────────────────────────────────────────
    BATCH_SIZE: int = 128
    EPOCHS: int = 30
    LR: float = 0.01
    L2_COEFF: float = 0.0001
    N_HIDDEN: int = 512       # hidden neurons in FC SNN

    # ── Derived ────────────────────────────────────────────────────────────────
    N_PIX: int = IMG_H * IMG_W        # 784
    POOL: int = 2                      # pooling stride in V1 L2/3
    POOLED: int = (IMG_H // POOL) * (IMG_W // POOL)   # 196

    # Feature vector length output by V1 L2/3 (sent to FC SNN)
    # Form: N_FORM channels × POOLED pixels
    # Color: N_COLOR channels × POOLED pixels
    N_FEATURES: int = (N_FORM + N_COLOR) * POOLED    # 11 × 196 = 2156


# ==============================================================================
# SECTION 3 — MNIST DATA PIPELINE (PyTorch)
# ==============================================================================
def clamp_01(x):
    return torch.clamp(x, 0.0, 1.0)

def get_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = Config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download and return MNIST train / test DataLoaders.

    Returns:
        train_loader: shuffled, batch_size=128
        test_loader:  not shuffled, batch_size=128
    """
    # Same normalisation used in paper (pixel values → roughly [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalise so the minimum becomes 0 and maximum ≈ 1
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(clamp_01),  # clip to [0,1]
    ])

    train_ds = datasets.MNIST(
        root=data_dir, train=True,  download=True, transform=transform
    )
    test_ds  = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"[MNIST] Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")
    return train_loader, test_loader


# ==============================================================================
# SECTION 4 — SPIKE ENCODING  (digit-spike transformation, paper §3.1)
# ==============================================================================

def image_to_spikes(
    img: np.ndarray,              # (H, W) float32, values in [0, 1]
    T: int = Config.T_WINDOW,
) -> np.ndarray:
    """
    Stochastic Poisson spike encoder (paper §3.1 steps 1-4).

    For each pixel i and time step t:
        r_i ~ Uniform(0, 1)
        s_i(t) = 1  if r_i >= (1 - v_i)  else 0

    Returns:
        spikes: (T, H*W) binary float32 array
    """
    v = img.flatten().astype(np.float32)
    v = np.clip(v, 0.0, 1.0)

    r = np.random.uniform(0.0, 1.0, size=(T, len(v))).astype(np.float32)
    return (r >= (1.0 - v[None, :])).astype(np.float32)   # (T, N_PIX)


def image_to_multimodal_spikes(
    image: torch.Tensor,          # (1, H, W) float32 normalised MNIST image
    T: int = Config.T_WINDOW,
) -> Dict[str, np.ndarray]:
    """
    Convert a grayscale MNIST image to 4 spike modalities:
      INRod : light intensity   (grayscale)
      INL   : long-wave  / L cone  (red-biased)
      INM   : medium-wave / M cone (green → same as gray for MNIST)
      INS   : short-wave / S cone  (blue-biased)

    Because MNIST is grayscale, the three cone channels are approximated
    by slight luminance rescaling (a common assumption in greyscale SNN
    models of the primary visual pathway).

    Returns dict of {channel_name: (T, N_PIX) float32 arrays}.
    """
    img = image.numpy().squeeze(0)   # (H, W)
    img = np.clip(img, 0.0, 1.0)

    rod  = img
    cL   = np.clip(img * 1.10, 0.0, 1.0)   # L slightly brighter
    cM   = img                               # M same as intensity
    cS   = np.clip(img * 0.90, 0.0, 1.0)   # S slightly dimmer

    return {
        "INRod": image_to_spikes(rod,  T),
        "INL":   image_to_spikes(cL,   T),
        "INM":   image_to_spikes(cM,   T),
        "INS":   image_to_spikes(cS,   T),
    }


# ==============================================================================
# SECTION 5 — RECEPTIVE-FIELD KERNEL BUILDERS
# ==============================================================================

def _grid(r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return meshgrid of shape (2r+1, 2r+1)."""
    sz = 2 * r + 1
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x.astype(np.float32), y.astype(np.float32)


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    """Normalised Gaussian kernel — used for noise-reduction circuit."""
    x, y = _grid(radius)
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def dog_kernel(radius: int, sigma_c: float, sigma_s: float,
               on_center: bool = True) -> np.ndarray:
    """
    Difference-of-Gaussians kernel for center-surround receptive fields.
    Used in ganglion cells (paper §2.1) and LGN (paper §2.2).
    """
    x, y = _grid(radius)
    gc = np.exp(-(x**2 + y**2) / (2 * sigma_c**2))
    gs = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
    gc /= gc.sum()
    gs /= gs.sum()
    k  = gc - gs if on_center else gs - gc
    return k.astype(np.float32)


def gabor_kernel(size: int, theta: float, sigma: float = 2.0,
                 freq: float = 0.45) -> np.ndarray:
    """
    Gabor filter — orientation-selective receptive fields in V1-L4 form pathway.
    Paper §2.3: 'post-synaptic neurons have Gabor-like receptive fields in V1 L4.'
    """
    r = size // 2
    x, y = _grid(r)
    x_rot =  x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
    carrier  = np.cos(2 * np.pi * freq * x_rot)
    k = envelope * carrier
    k -= k.mean()
    norm = np.abs(k).sum()
    return (k / (norm + 1e-8)).astype(np.float32)


def double_opponent_kernel(radius: int, center_excit: bool = True) -> np.ndarray:
    """
    Double-opponent kernel for V1-L4 colour channels.
    Paper §2.3: 'neurons in these three channels all have double-opponent
    receptive fields.'
    """
    return dog_kernel(radius, sigma_c=0.5, sigma_s=1.5,
                      on_center=center_excit)


def rf_weight_matrix(H: int, W: int, kernel: np.ndarray) -> np.ndarray:
    """
    Expand a 2-D kernel into a full (H*W, H*W) weight matrix that applies
    the same local receptive field to every pixel (shared weights).
    Zero-padding is used at the border.

    Returns:
        W_mat: (n_out=H*W, n_in=H*W) float32 ndarray
    """
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    n = H * W
    W_mat = np.zeros((n, n), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            out_idx = i * W + j
            for ki in range(kH):
                for kj in range(kW):
                    ii = i + ki - pH
                    jj = j + kj - pW
                    if 0 <= ii < H and 0 <= jj < W:
                        in_idx = ii * W + jj
                        W_mat[out_idx, in_idx] = kernel[ki, kj]
    return W_mat


# ==============================================================================
# SECTION 6 — CUSTOM LAVA PROCESSES
# ==============================================================================

# ─────────────────────────────────────────────────────────────────
#  6-A  GradedNeuron  (photoreceptor, paper Eq. 2)
# ─────────────────────────────────────────────────────────────────

class GradedNeuron(AbstractProcess):
    """
    Grade-neuron model (Eq. 2):
        τ dV/dt = Iin·Rm + Vrest − V
        S(t) = max(0, tanh((V − Ep) / Vsl))

    Output is a continuous value (graded) passed to downstream synapses
    via tanh activation (matches Eq. 4 synapse model in the paper).
    """

    def __init__(self, n: int, tau: float, rm: float, vrest: float,
                 ep: float, vsl: float, **kwargs):
        super().__init__(n=n, tau=tau, rm=rm, vrest=vrest, ep=ep, vsl=vsl,
                         **kwargs)
        self.a_in  = InPort(shape=(n,))
        self.s_out = OutPort(shape=(n,))

        self.v      = Var(shape=(n,), init=np.full(n, vrest, dtype=np.float32))
        self.tau    = Var(shape=(1,), init=np.array([tau],    dtype=np.float32))
        self.rm     = Var(shape=(1,), init=np.array([rm],     dtype=np.float32))
        self.vrest  = Var(shape=(1,), init=np.array([vrest],  dtype=np.float32))
        self.ep     = Var(shape=(1,), init=np.array([ep],     dtype=np.float32))
        self.vsl    = Var(shape=(1,), init=np.array([vsl],    dtype=np.float32))


@implements(proc=GradedNeuron, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class GradedNeuronModel(PyLoihiProcessModel):

    a_in  : PyInPort  = LavaPyType(PyInPort.VEC_DENSE,  float)
    s_out : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    v     : np.ndarray = LavaPyType(np.ndarray, float)
    tau   : np.ndarray = LavaPyType(np.ndarray, float)
    rm    : np.ndarray = LavaPyType(np.ndarray, float)
    vrest : np.ndarray = LavaPyType(np.ndarray, float)
    ep    : np.ndarray = LavaPyType(np.ndarray, float)
    vsl   : np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        a   = self.a_in.recv()
        tau, rm, vr = float(self.tau[0]), float(self.rm[0]), float(self.vrest[0])
        ep,  vsl    = float(self.ep[0]),  float(self.vsl[0])

        # Euler integration: τ dV/dt = a·Rm + Vrest − V
        self.v += (a * rm + vr - self.v) / tau * Config.DT

        # Graded output  (Eq. 4 tanh activation)
        s = np.maximum(0.0, np.tanh((self.v - ep) / vsl))
        self.s_out.send(s.astype(np.float32))


# ─────────────────────────────────────────────────────────────────
#  6-B  CustomLIF  (all non-graded layers, paper Eq. 1)
# ─────────────────────────────────────────────────────────────────

class CustomLIF(AbstractProcess):
    """
    Leaky Integrate-and-Fire neuron (Eq. 1):
        τ dV/dt = Iin·Rm + Vrest − V
        Fire when V ≥ Vth → S(t) = 1,  V reset to Vrest
    """

    def __init__(self, n: int, tau: float, rm: float, vrest: float,
                 vth: float = Config.VTH, **kwargs):
        super().__init__(n=n, tau=tau, rm=rm, vrest=vrest, vth=vth, **kwargs)
        self.a_in  = InPort(shape=(n,))
        self.s_out = OutPort(shape=(n,))

        self.v     = Var(shape=(n,), init=np.full(n, vrest, dtype=np.float32))
        self.tau   = Var(shape=(1,), init=np.array([tau],   dtype=np.float32))
        self.rm    = Var(shape=(1,), init=np.array([rm],    dtype=np.float32))
        self.vrest = Var(shape=(1,), init=np.array([vrest], dtype=np.float32))
        self.vth   = Var(shape=(1,), init=np.array([vth],   dtype=np.float32))


@implements(proc=CustomLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class CustomLIFModel(PyLoihiProcessModel):

    a_in  : PyInPort  = LavaPyType(PyInPort.VEC_DENSE,  float)
    s_out : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    v     : np.ndarray = LavaPyType(np.ndarray, float)
    tau   : np.ndarray = LavaPyType(np.ndarray, float)
    rm    : np.ndarray = LavaPyType(np.ndarray, float)
    vrest : np.ndarray = LavaPyType(np.ndarray, float)
    vth   : np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        a  = self.a_in.recv()
        tau, rm = float(self.tau[0]),   float(self.rm[0])
        vr, vth = float(self.vrest[0]), float(self.vth[0])

        # Euler integration
        self.v += (a * rm + vr - self.v) / tau * Config.DT

        # Threshold-and-reset
        spk = (self.v >= vth).astype(np.float32)
        self.v[spk > 0] = vr

        self.s_out.send(spk)


# ─────────────────────────────────────────────────────────────────
#  6-C  STDPDense  (trainable Dense connection with STDP, Eq. 5-6)
# ─────────────────────────────────────────────────────────────────

class STDPDense(AbstractProcess):
    """
    Dense synaptic connection trained with additive STDP (Eq. 5-6).
    Nearest-neighbour scheme: eligibility traces reset to 1 on each spike.

    Ports:
        s_in      : pre-synaptic spikes
        a_out     : post-synaptic current
        s_post_in : post-synaptic spikes (fed back for STDP)
    """

    def __init__(self, weights: np.ndarray, **kwargs):
        n_out, n_in = weights.shape
        super().__init__(weights=weights, n_in=n_in, n_out=n_out, **kwargs)

        self.s_in      = InPort(shape=(n_in,))
        self.a_out     = OutPort(shape=(n_out,))
        self.s_post_in = InPort(shape=(n_out,))   # feedback from post neurons

        self.weights    = Var(shape=(n_out, n_in), init=weights.astype(np.float32))
        self.trace_pre  = Var(shape=(n_in,),       init=np.zeros(n_in,  np.float32))
        self.trace_post = Var(shape=(n_out,),      init=np.zeros(n_out, np.float32))

        # STDP hyper-parameters as Vars (read-only scalars stored as length-1 arrays)
        self.tau_plus  = Var(shape=(1,), init=np.array([Config.TAU_PLUS],  np.float32))
        self.tau_minus = Var(shape=(1,), init=np.array([Config.TAU_MINUS], np.float32))
        self.a_pre     = Var(shape=(1,), init=np.array([Config.A_PRE],     np.float32))
        self.a_post    = Var(shape=(1,), init=np.array([Config.A_POST],    np.float32))
        self.w_min     = Var(shape=(1,), init=np.array([Config.W_MIN],     np.float32))
        self.w_max     = Var(shape=(1,), init=np.array([Config.W_MAX],     np.float32))


@implements(proc=STDPDense, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class STDPDenseModel(PyLoihiProcessModel):

    s_in      : PyInPort  = LavaPyType(PyInPort.VEC_DENSE,  float)
    a_out     : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    s_post_in : PyInPort  = LavaPyType(PyInPort.VEC_DENSE,  float)

    weights    : np.ndarray = LavaPyType(np.ndarray, float)
    trace_pre  : np.ndarray = LavaPyType(np.ndarray, float)
    trace_post : np.ndarray = LavaPyType(np.ndarray, float)
    tau_plus   : np.ndarray = LavaPyType(np.ndarray, float)
    tau_minus  : np.ndarray = LavaPyType(np.ndarray, float)
    a_pre      : np.ndarray = LavaPyType(np.ndarray, float)
    a_post     : np.ndarray = LavaPyType(np.ndarray, float)
    w_min      : np.ndarray = LavaPyType(np.ndarray, float)
    w_max      : np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_pre  = self.s_in.recv()
        s_post = self.s_post_in.recv()

        dt   = Config.DT
        tp   = float(self.tau_plus[0])
        tm   = float(self.tau_minus[0])
        ap   = float(self.a_post[0])
        am   = float(self.a_pre[0])
        wmin = float(self.w_min[0])
        wmax = float(self.w_max[0])

        # ── Decay eligibility traces ──────────────────────────────────────────
        self.trace_pre  *= np.exp(-dt / tp)
        self.trace_post *= np.exp(-dt / tm)

        # ── Nearest-neighbour: reset trace to 1.0 on spike ───────────────────
        self.trace_pre [s_pre  > 0] = 1.0
        self.trace_post[s_post > 0] = 1.0

        # ── Additive STDP weight update (Eq. 5-6) ────────────────────────────
        # LTP : post fires → correlate with pre trace  → Δw = A_post × x_pre
        # LTD : pre  fires → correlate with post trace → Δw = −A_pre  × x_post
        dw_ltp = ap * np.outer(s_post, self.trace_pre)   # (n_out, n_in)
        dw_ltd = am * np.outer(self.trace_post, s_pre)   # (n_out, n_in)
        self.weights = np.clip(self.weights + dw_ltp - dw_ltd, wmin, wmax)

        # ── Forward pass ─────────────────────────────────────────────────────
        self.a_out.send(self.weights @ s_pre)


# ─────────────────────────────────────────────────────────────────
#  6-D  SpikeInput  — injects external spike patterns each step
# ─────────────────────────────────────────────────────────────────

class SpikeInput(AbstractProcess):
    """Injects a pre-computed spike vector into the network each time step."""

    def __init__(self, n: int, **kwargs):
        super().__init__(n=n, **kwargs)
        self.s_out  = OutPort(shape=(n,))
        self.spikes = Var(shape=(n,), init=np.zeros(n, dtype=np.float32))


@implements(proc=SpikeInput, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class SpikeInputModel(PyLoihiProcessModel):

    s_out  : PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    spikes : np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        self.s_out.send(self.spikes.astype(np.float32))


# ─────────────────────────────────────────────────────────────────
#  6-E  SpikeCounter  — accumulates spike counts over T steps
# ─────────────────────────────────────────────────────────────────

class SpikeCounter(AbstractProcess):
    """Accumulates spike counts; read .spike_counts after RunSteps(T)."""

    def __init__(self, n: int, **kwargs):
        super().__init__(n=n, **kwargs)
        self.s_in        = InPort(shape=(n,))
        self.spike_counts = Var(shape=(n,), init=np.zeros(n, dtype=np.float32))


@implements(proc=SpikeCounter, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class SpikeCounterModel(PyLoihiProcessModel):

    s_in         : PyInPort  = LavaPyType(PyInPort.VEC_DENSE, float)
    spike_counts : np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        self.spike_counts += self.s_in.recv()


# ==============================================================================
# SECTION 7 — RETINA SUB-NETWORK  (paper §2.1)
# ==============================================================================

class RetinaSubNetwork:
    """
    Simulates the mammalian retina:

    Layers:
        Photoreceptor (graded)   — Rod  +  ConeL / ConeM / ConeS
        Bipolar Cell   (LIF)     — RBC  +  BC[LMS][on|off]
        Ganglion Cell  (LIF)     — GC[LMS][on|off]

    Key circuits:
        • Noise-reduction     : Gaussian lateral inhibition at photoreceptor
        • Light-adaptation    : RBC → GC inhibitory rod pathway (Eq. rod pathway)
        • Center-surround DoG : GC layer edge enhancement
    """

    def __init__(self):
        H, W, n = Config.IMG_H, Config.IMG_W, Config.N_PIX

        # ── Input injectors ──────────────────────────────────────────────────
        self.inp_rod = SpikeInput(n=n, name="INRod")
        self.inp_L   = SpikeInput(n=n, name="INL")
        self.inp_M   = SpikeInput(n=n, name="INM")
        self.inp_S   = SpikeInput(n=n, name="INS")

        # ── Photoreceptor layer ───────────────────────────────────────────────
        gp = dict(tau=Config.TAU_GRADE, rm=Config.RM_GRADE,
                  vrest=Config.VREST_GRADE, ep=Config.EP, vsl=Config.VSL)
        self.rod    = GradedNeuron(n=n, **gp, name="Rod")
        self.cone_L = GradedNeuron(n=n, **gp, name="ConeL")
        self.cone_M = GradedNeuron(n=n, **gp, name="ConeM")
        self.cone_S = GradedNeuron(n=n, **gp, name="ConeS")

        # ── Bipolar cell layer ────────────────────────────────────────────────
        bc_p = dict(tau=Config.TAU_BC, rm=Config.RM_BC,
                    vrest=Config.VREST_BC, vth=Config.VTH)
        self.rbc     = CustomLIF(n=n, **bc_p, name="RBC")
        self.bc_L_on  = CustomLIF(n=n, **bc_p, name="BCLon")
        self.bc_L_off = CustomLIF(n=n, **bc_p, name="BCLoff")
        self.bc_M_on  = CustomLIF(n=n, **bc_p, name="BCMon")
        self.bc_M_off = CustomLIF(n=n, **bc_p, name="BCMoff")
        self.bc_S_on  = CustomLIF(n=n, **bc_p, name="BCSon")
        self.bc_S_off = CustomLIF(n=n, **bc_p, name="BCSoff")

        # ── Ganglion cell layer ───────────────────────────────────────────────
        gc_p = dict(tau=Config.TAU_LIF, rm=Config.RM_LIF,
                    vrest=Config.VREST_LIF, vth=Config.VTH)
        self.gc_L_on  = CustomLIF(n=n, **gc_p, name="GCLon")
        self.gc_L_off = CustomLIF(n=n, **gc_p, name="GCLoff")
        self.gc_M_on  = CustomLIF(n=n, **gc_p, name="GCMon")
        self.gc_M_off = CustomLIF(n=n, **gc_p, name="GCMoff")
        self.gc_S_on  = CustomLIF(n=n, **gc_p, name="GCSon")
        self.gc_S_off = CustomLIF(n=n, **gc_p, name="GCSoff")

        self._build_synapses(H, W, n)
        self._wire()

    # ── Synapse builders ──────────────────────────────────────────────────────

    def _build_synapses(self, H: int, W: int, n: int):
        """Create all weight matrices and Dense/STDPDense processes."""

        # ── Noise-reduction: Gaussian RF at Rod → noise smoothing ────────────
        gauss     = gaussian_kernel(Config.RF_RBC, sigma=1.5)
        W_rod_rbc = rf_weight_matrix(H, W, gauss)
        self.syn_rod_rbc = STDPDense(weights=W_rod_rbc, name="syn_rod_rbc")

        # ── Cone → BC (one-to-one, ±identity) ────────────────────────────────
        # BCoff = excitatory (positive I → depolarises)
        # BCon  = inhibitory cone input (centre opponent; BCs already ON due to sign flip)
        eye    = np.eye(n, dtype=np.float32)
        neg_eye = -eye

        self.syn_cL_bcLoff = Dense(weights= eye,    name="syn_cL_bcLoff")
        self.syn_cL_bcLon  = Dense(weights= neg_eye, name="syn_cL_bcLon")
        self.syn_cM_bcMoff = Dense(weights= eye,    name="syn_cM_bcMoff")
        self.syn_cM_bcMon  = Dense(weights= neg_eye, name="syn_cM_bcMon")
        self.syn_cS_bcSoff = Dense(weights= eye,    name="syn_cS_bcSoff")
        self.syn_cS_bcSon  = Dense(weights= neg_eye, name="syn_cS_bcSon")

        # ── GC layer: center-surround DoG receptive fields ───────────────────
        r   = Config.RF_GC_INH
        dog_on  = dog_kernel(r, sigma_c=0.5, sigma_s=1.5, on_center=True)
        dog_off = dog_kernel(r, sigma_c=0.5, sigma_s=1.5, on_center=False)

        W_on  = rf_weight_matrix(H, W, dog_on)
        W_off = rf_weight_matrix(H, W, dog_off)

        # Separate excitatory (+) and inhibitory (−) parts
        W_on_exc  = np.maximum(0, W_on)
        W_on_inh  = np.minimum(0, W_on)
        W_off_exc = np.maximum(0, W_off)
        W_off_inh = np.minimum(0, W_off)

        # BCon → GCon (excitatory center, learnable via STDP)
        self.syn_bcLon_gcLon   = STDPDense(weights=W_on_exc, name="bcLon_gcLon")
        self.syn_bcLoff_gcLon  = Dense(weights=W_on_inh,     name="bcLoff_gcLon")
        self.syn_bcMon_gcMon   = STDPDense(weights=W_on_exc, name="bcMon_gcMon")
        self.syn_bcMoff_gcMon  = Dense(weights=W_on_inh,     name="bcMoff_gcMon")
        self.syn_bcSon_gcSon   = STDPDense(weights=W_on_exc, name="bcSon_gcSon")
        self.syn_bcSoff_gcSon  = Dense(weights=W_on_inh,     name="bcSoff_gcSon")

        # BCoff → GCoff
        self.syn_bcLoff_gcLoff = STDPDense(weights=W_off_exc, name="bcLoff_gcLoff")
        self.syn_bcLon_gcLoff  = Dense(weights=W_off_inh,     name="bcLon_gcLoff")
        self.syn_bcMoff_gcMoff = STDPDense(weights=W_off_exc, name="bcMoff_gcMoff")
        self.syn_bcMon_gcMoff  = Dense(weights=W_off_inh,     name="bcMon_gcMoff")
        self.syn_bcSoff_gcSoff = STDPDense(weights=W_off_exc, name="bcSoff_gcSoff")
        self.syn_bcSon_gcSoff  = Dense(weights=W_off_inh,     name="bcSon_gcSoff")

        # ── Light-adaptation: RBC → GC (inhibitory, radius=3) ────────────────
        W_rbc_gc = -rf_weight_matrix(H, W, gauss)   # inhibitory
        self.syn_rbc_gcLon = Dense(weights=W_rbc_gc, name="rbc_gcLon")
        self.syn_rbc_gcMon = Dense(weights=W_rbc_gc, name="rbc_gcMon")
        self.syn_rbc_gcSon = Dense(weights=W_rbc_gc, name="rbc_gcSon")

    def _wire(self):
        """Connect all processes in the retina sub-network."""

        # ── Input → photoreceptor ─────────────────────────────────────────────
        self.inp_rod.s_out.connect(self.rod.a_in)
        self.inp_L.s_out.connect(self.cone_L.a_in)
        self.inp_M.s_out.connect(self.cone_M.a_in)
        self.inp_S.s_out.connect(self.cone_S.a_in)

        # ── Rod → RBC ─────────────────────────────────────────────────────────
        self.rod.s_out.connect(self.syn_rod_rbc.s_in)
        # (STDP post-port wired after GC created — see _wire_stdp_feedback)
        self.syn_rod_rbc.a_out.connect(self.rbc.a_in)

        # ── Cone → BC (one-to-one) ────────────────────────────────────────────
        self.cone_L.s_out.connect(self.syn_cL_bcLoff.s_in)
        self.syn_cL_bcLoff.a_out.connect(self.bc_L_off.a_in)

        self.cone_L.s_out.connect(self.syn_cL_bcLon.s_in)
        self.syn_cL_bcLon.a_out.connect(self.bc_L_on.a_in)

        self.cone_M.s_out.connect(self.syn_cM_bcMoff.s_in)
        self.syn_cM_bcMoff.a_out.connect(self.bc_M_off.a_in)

        self.cone_M.s_out.connect(self.syn_cM_bcMon.s_in)
        self.syn_cM_bcMon.a_out.connect(self.bc_M_on.a_in)

        self.cone_S.s_out.connect(self.syn_cS_bcSoff.s_in)
        self.syn_cS_bcSoff.a_out.connect(self.bc_S_off.a_in)

        self.cone_S.s_out.connect(self.syn_cS_bcSon.s_in)
        self.syn_cS_bcSon.a_out.connect(self.bc_S_on.a_in)

        # ── BC → GC (center-surround) ─────────────────────────────────────────
        self.bc_L_on.s_out.connect(self.syn_bcLon_gcLon.s_in)
        self.bc_L_off.s_out.connect(self.syn_bcLoff_gcLon.s_in)
        self.syn_bcLon_gcLon.a_out.connect(self.gc_L_on.a_in)
        self.syn_bcLoff_gcLon.a_out.connect(self.gc_L_on.a_in)

        self.bc_M_on.s_out.connect(self.syn_bcMon_gcMon.s_in)
        self.bc_M_off.s_out.connect(self.syn_bcMoff_gcMon.s_in)
        self.syn_bcMon_gcMon.a_out.connect(self.gc_M_on.a_in)
        self.syn_bcMoff_gcMon.a_out.connect(self.gc_M_on.a_in)

        self.bc_S_on.s_out.connect(self.syn_bcSon_gcSon.s_in)
        self.bc_S_off.s_out.connect(self.syn_bcSoff_gcSon.s_in)
        self.syn_bcSon_gcSon.a_out.connect(self.gc_S_on.a_in)
        self.syn_bcSoff_gcSon.a_out.connect(self.gc_S_on.a_in)

        self.bc_L_off.s_out.connect(self.syn_bcLoff_gcLoff.s_in)
        self.bc_L_on.s_out.connect(self.syn_bcLon_gcLoff.s_in)
        self.syn_bcLoff_gcLoff.a_out.connect(self.gc_L_off.a_in)
        self.syn_bcLon_gcLoff.a_out.connect(self.gc_L_off.a_in)

        self.bc_M_off.s_out.connect(self.syn_bcMoff_gcMoff.s_in)
        self.bc_M_on.s_out.connect(self.syn_bcMon_gcMoff.s_in)
        self.syn_bcMoff_gcMoff.a_out.connect(self.gc_M_off.a_in)
        self.syn_bcMon_gcMoff.a_out.connect(self.gc_M_off.a_in)

        self.bc_S_off.s_out.connect(self.syn_bcSoff_gcSoff.s_in)
        self.bc_S_on.s_out.connect(self.syn_bcSon_gcSoff.s_in)
        self.syn_bcSoff_gcSoff.a_out.connect(self.gc_S_off.a_in)
        self.syn_bcSon_gcSoff.a_out.connect(self.gc_S_off.a_in)

        # ── STDP feedback: GCon → STDP post port ─────────────────────────────
        self.gc_L_on.s_out.connect(self.syn_bcLon_gcLon.s_post_in)
        self.gc_M_on.s_out.connect(self.syn_bcMon_gcMon.s_post_in)
        self.gc_S_on.s_out.connect(self.syn_bcSon_gcSon.s_post_in)
        self.gc_L_off.s_out.connect(self.syn_bcLoff_gcLoff.s_post_in)
        self.gc_M_off.s_out.connect(self.syn_bcMoff_gcMoff.s_post_in)
        self.gc_S_off.s_out.connect(self.syn_bcSoff_gcSoff.s_post_in)
        # RBC STDP feedback via rbc → rbc (self-consistency)
        self.rbc.s_out.connect(self.syn_rod_rbc.s_post_in)

        # ── RBC → GC (light-adaptation, inhibitory) ──────────────────────────
        self.rbc.s_out.connect(self.syn_rbc_gcLon.s_in)
        self.syn_rbc_gcLon.a_out.connect(self.gc_L_on.a_in)

        self.rbc.s_out.connect(self.syn_rbc_gcMon.s_in)
        self.syn_rbc_gcMon.a_out.connect(self.gc_M_on.a_in)

        self.rbc.s_out.connect(self.syn_rbc_gcSon.s_in)
        self.syn_rbc_gcSon.a_out.connect(self.gc_S_on.a_in)

    # ── GC output nodes (used by LGN sub-network) ────────────────────────────

    @property
    def gc_on_ports(self):
        """[GCLon, GCMon, GCSon] s_out ports → LGN input."""
        return [self.gc_L_on.s_out, self.gc_M_on.s_out, self.gc_S_on.s_out]

    @property
    def gc_off_ports(self):
        """[GCLoff, GCMoff, GCSoff] s_out ports → LGN inhibitory."""
        return [self.gc_L_off.s_out, self.gc_M_off.s_out, self.gc_S_off.s_out]


# ==============================================================================
# SECTION 8 — LGN SUB-NETWORK  (paper §2.2)
# ==============================================================================

class LGNSubNetwork:
    """
    Lateral Geniculate Nucleus:

    Layers:
        Input layer  (LIF): LGN3/4p, LGN5/6p, LGNKp
                            Receives from GCon (excitatory) and GCoff (inhibitory)
        Output layer (LIF): LGN3/4,  LGN5/6,  LGNK
                            One-to-one from Input; adjusted by V1-L6 feedback

    Three channels correspond to parvocellular (3/4, 5/6) and
    koniocellular (K) divisions of biological LGN.
    """

    def __init__(self, retina: RetinaSubNetwork):
        H, W, n = Config.IMG_H, Config.IMG_W, Config.N_PIX
        lif_p = dict(tau=Config.TAU_LIF, rm=Config.RM_LIF,
                     vrest=Config.VREST_LIF, vth=Config.VTH)

        # ── Input layer (LGN p-channels) ─────────────────────────────────────
        self.lgn_34p = CustomLIF(n=n, **lif_p, name="LGN34p")
        self.lgn_56p = CustomLIF(n=n, **lif_p, name="LGN56p")
        self.lgn_Kp  = CustomLIF(n=n, **lif_p, name="LGNKp")

        # ── Output layer ─────────────────────────────────────────────────────
        self.lgn_34  = CustomLIF(n=n, **lif_p, name="LGN34")
        self.lgn_56  = CustomLIF(n=n, **lif_p, name="LGN56")
        self.lgn_K   = CustomLIF(n=n, **lif_p, name="LGNK")

        self._build_synapses(H, W, n)
        self._wire(retina)

    def _build_synapses(self, H: int, W: int, n: int):
        eye     = np.eye(n, dtype=np.float32)
        # Inhibitory RF from GCoff → LGN input (radius=1 surround)
        r         = Config.RF_LGN_INH
        dog_inh   = dog_kernel(r, sigma_c=0.5, sigma_s=1.5, on_center=False)
        W_inh     = rf_weight_matrix(H, W, dog_inh)
        W_inh_neg = np.minimum(0, W_inh)   # keep only negative part

        # GCon → LGNp (excitatory, 1:1)
        self.syn_gcLon_34p = Dense(weights= eye,      name="gcLon_34p")
        self.syn_gcMon_56p = Dense(weights= eye,      name="gcMon_56p")
        self.syn_gcSon_Kp  = Dense(weights= eye,      name="gcSon_Kp")

        # GCoff → LGNp (inhibitory surround)
        self.syn_gcLoff_34p = Dense(weights=W_inh_neg, name="gcLoff_34p")
        self.syn_gcMoff_56p = Dense(weights=W_inh_neg, name="gcMoff_56p")
        self.syn_gcSoff_Kp  = Dense(weights=W_inh_neg, name="gcSoff_Kp")

        # LGNp → LGN output (1:1 excitatory, STDP)
        self.syn_34p_34 = STDPDense(weights=eye, name="34p_34")
        self.syn_56p_56 = STDPDense(weights=eye, name="56p_56")
        self.syn_Kp_K   = STDPDense(weights=eye, name="Kp_K")

    def _wire(self, retina: RetinaSubNetwork):
        gc_on  = retina.gc_on_ports    # [gcL, gcM, gcS]
        gc_off = retina.gc_off_ports

        # GCon → LGN input (excitatory)
        gc_on[0].connect(self.syn_gcLon_34p.s_in)
        self.syn_gcLon_34p.a_out.connect(self.lgn_34p.a_in)

        gc_on[1].connect(self.syn_gcMon_56p.s_in)
        self.syn_gcMon_56p.a_out.connect(self.lgn_56p.a_in)

        gc_on[2].connect(self.syn_gcSon_Kp.s_in)
        self.syn_gcSon_Kp.a_out.connect(self.lgn_Kp.a_in)

        # GCoff → LGN input (inhibitory)
        gc_off[0].connect(self.syn_gcLoff_34p.s_in)
        self.syn_gcLoff_34p.a_out.connect(self.lgn_34p.a_in)

        gc_off[1].connect(self.syn_gcMoff_56p.s_in)
        self.syn_gcMoff_56p.a_out.connect(self.lgn_56p.a_in)

        gc_off[2].connect(self.syn_gcSoff_Kp.s_in)
        self.syn_gcSoff_Kp.a_out.connect(self.lgn_Kp.a_in)

        # LGNp → LGN output
        self.lgn_34p.s_out.connect(self.syn_34p_34.s_in)
        self.syn_34p_34.a_out.connect(self.lgn_34.a_in)
        self.lgn_34.s_out.connect(self.syn_34p_34.s_post_in)   # STDP feedback

        self.lgn_56p.s_out.connect(self.syn_56p_56.s_in)
        self.syn_56p_56.a_out.connect(self.lgn_56.a_in)
        self.lgn_56.s_out.connect(self.syn_56p_56.s_post_in)

        self.lgn_Kp.s_out.connect(self.syn_Kp_K.s_in)
        self.syn_Kp_K.a_out.connect(self.lgn_K.a_in)
        self.lgn_K.s_out.connect(self.syn_Kp_K.s_post_in)

    def wire_feedback(self, v1_l6_ports: List):
        """
        Attach V1-L6 → LGN output feedback connections (paper §2.2 and §2.3).
        Called after V1SubNetwork is instantiated.
        """
        n = Config.N_PIX
        eye = np.eye(n, dtype=np.float32)

        self.syn_v1l6_34 = Dense(weights=eye, name="v1l6_34")
        self.syn_v1l6_56 = Dense(weights=eye, name="v1l6_56")
        self.syn_v1l6_K  = Dense(weights=eye, name="v1l6_K")

        v1_l6_ports[0].connect(self.syn_v1l6_34.s_in)
        self.syn_v1l6_34.a_out.connect(self.lgn_34.a_in)

        v1_l6_ports[1].connect(self.syn_v1l6_56.s_in)
        self.syn_v1l6_56.a_out.connect(self.lgn_56.a_in)

        v1_l6_ports[2].connect(self.syn_v1l6_K.s_in)
        self.syn_v1l6_K.a_out.connect(self.lgn_K.a_in)

    @property
    def output_ports(self):
        """[lgn_34, lgn_56, lgn_K] s_out ports → V1."""
        return [self.lgn_34.s_out, self.lgn_56.s_out, self.lgn_K.s_out]


# ==============================================================================
# SECTION 9 — V1 SUB-NETWORK  (paper §2.3)
# ==============================================================================

class V1SubNetwork:
    """
    Primary Visual Cortex (V1):

    Layers:
        L6  (simple cells, LIF)  : 3 channels, 1:1 from LGN, feedback to LGN
        L4  (simple cells, LIF)  : N_FORM form channels (Gabor RF)
                                   N_COLOR colour channels (double-opponent RF)
                                   Each + inhibitory lateral channel
        L2/3 (complex cells, LIF): Pool + combine L4 features
                                   N_FORM form channels, N_COLOR colour channels

    Output: V1 L2/3 spike counts → FC SNN feature vector
    """

    def __init__(self, lgn: LGNSubNetwork):
        H, W, n = Config.IMG_H, Config.IMG_W, Config.N_PIX
        nF, nC  = Config.N_FORM, Config.N_COLOR
        lif_p   = dict(tau=Config.TAU_LIF, rm=Config.RM_LIF,
                       vrest=Config.VREST_LIF, vth=Config.VTH)

        # ── V1 L6 : 3 channels, 1 neuron / pixel ─────────────────────────────
        self.l6 = [CustomLIF(n=n, **lif_p, name=f"V1L6_ch{i}") for i in range(3)]

        # ── V1 L4 form pathway : nF channels + nF inhibitory ─────────────────
        self.l4_form     = [CustomLIF(n=n, **lif_p, name=f"V1L4form{i}")    for i in range(nF)]
        self.l4_form_inh = [CustomLIF(n=n, **lif_p, name=f"V1L4formInh{i}") for i in range(nF)]

        # ── V1 L4 colour pathway : nC channels + nC inhibitory ───────────────
        self.l4_color     = [CustomLIF(n=n, **lif_p, name=f"V1L4color{i}")    for i in range(nC)]
        self.l4_color_inh = [CustomLIF(n=n, **lif_p, name=f"V1L4colorInh{i}") for i in range(nC)]

        # ── V1 L2/3 : pooled, nF form + nC colour ────────────────────────────
        n_pool = Config.POOLED
        self.l23_form     = [CustomLIF(n=n_pool, **lif_p, name=f"V1L23form{i}")    for i in range(nF)]
        self.l23_form_inh = [CustomLIF(n=n_pool, **lif_p, name=f"V1L23formInh{i}") for i in range(nF)]
        self.l23_color    = [CustomLIF(n=n_pool, **lif_p, name=f"V1L23color{i}")   for i in range(nC)]
        self.l23_color_inh= [CustomLIF(n=n_pool, **lif_p, name=f"V1L23colorInh{i}") for i in range(nC)]

        # ── Output spike counters (record feature vectors) ────────────────────
        self.counters_form  = [SpikeCounter(n=n_pool, name=f"cnt_form{i}")  for i in range(nF)]
        self.counters_color = [SpikeCounter(n=n_pool, name=f"cnt_color{i}") for i in range(nC)]

        self._build_synapses(H, W, n, n_pool, nF, nC)
        self._wire(lgn, H, W, n, n_pool, nF, nC)

    def _build_synapses(self, H, W, n, n_pool, nF, nC):
        """
        Construct weight matrices for:
          - LGN → V1 L6 (1:1)
          - LGN → V1 L4 form  (Gabor kernels)
          - LGN → V1 L4 color (double-opponent kernels)
          - Lateral inhibition (1:1 for inhibitory channels)
          - L4 → L2/3 pooling + combination
        """
        eye_n    = np.eye(n, dtype=np.float32)
        eye_pool = np.eye(n_pool, dtype=np.float32)

        # ── L6: one-to-one from each LGN output channel ──────────────────────
        self.syn_lgn_l6 = [Dense(weights=eye_n, name=f"lgn_l6_{i}") for i in range(3)]

        # ── L4 form: Gabor kernels at N_FORM orientations ────────────────────
        orientations = np.linspace(0, np.pi, nF, endpoint=False)
        gabor_weights = []
        for theta in orientations:
            k  = gabor_kernel(Config.GABOR_RF, theta=theta, sigma=2.0, freq=0.45)
            Wg = rf_weight_matrix(H, W, k)
            # LGN combined (sum of three channels) → form channel
            # Use absolute value positive part only → excitatory synapse
            gabor_weights.append(np.maximum(0, Wg))

        # Each L4 form channel receives from combined LGN (all 3 channels summed)
        # Implemented as three separate connections scaled by 1/3
        self.syn_lgn_l4form = []
        for i, Wg in enumerate(gabor_weights):
            syns_i = [STDPDense(weights=Wg / 3.0, name=f"lgn_l4form_{i}_ch{c}")
                      for c in range(3)]
            self.syn_lgn_l4form.append(syns_i)

        # ── L4 form lateral inhibition (1:1 excitatory to inh neuron) ────────
        self.syn_l4form_inh  = [Dense(weights= eye_n, name=f"l4f_inh_{i}")  for i in range(nF)]
        self.syn_inh_l4form  = [Dense(weights=-eye_n, name=f"inh_l4f_{i}")  for i in range(nF)]

        # ── L4 colour: double-opponent RFs ────────────────────────────────────
        # Ch 0: LGN34 center-excit / surround-inhib
        # Ch 1: LGN56 center-inhib / surround-excit
        # Ch 2: combined LGN34+56+K
        r_c = Config.RF_V1 + 1
        dop_excit = rf_weight_matrix(H, W, double_opponent_kernel(r_c, center_excit=True))
        dop_inhib = rf_weight_matrix(H, W, double_opponent_kernel(r_c, center_excit=False))

        W_c0 = np.maximum(0, dop_excit)    # center excitatory
        W_c1 = np.maximum(0, dop_inhib)    # center inhibitory (from LGN56)
        W_c2 = np.maximum(0, dop_excit) / 3.0   # combined

        self.syn_lgn_l4color = [
            [Dense(weights=W_c0, name="lgn34_color0")],                  # ch0 ← lgn34
            [Dense(weights=W_c1, name="lgn56_color1")],                  # ch1 ← lgn56
            [Dense(weights=W_c2, name="lgn34_color2_34"),                # ch2 ← lgn34,56,K
             Dense(weights=W_c2, name="lgn56_color2_56"),
             Dense(weights=W_c2, name="lgnK_color2_K")],
        ]

        # ── L4 colour lateral inhibition ─────────────────────────────────────
        self.syn_l4color_inh = [Dense(weights= eye_n, name=f"l4c_inh_{i}") for i in range(nC)]
        self.syn_inh_l4color = [Dense(weights=-eye_n, name=f"inh_l4c_{i}") for i in range(nC)]

        # ── L4 → L2/3 pooling (average pooling → dense) ──────────────────────
        #  Pooling matrix: (n_pool, n) where n_pool = (H/pool)*(W/pool)
        pool = Config.POOL
        W_pool = np.zeros((n_pool, n), dtype=np.float32)
        for pi in range(H // pool):
            for pj in range(W // pool):
                out_idx = pi * (W // pool) + pj
                for di in range(pool):
                    for dj in range(pool):
                        in_idx = (pi * pool + di) * W + (pj * pool + dj)
                        W_pool[out_idx, in_idx] = 1.0 / (pool * pool)

        # ── L4 form → L2/3 form  (each L2/3 channel combines all L4 form) ───
        # Full combination: (n_pool, n*nF) split into nF separate Dense processes
        self.syn_l4form_l23form = []   # [form_ch_i][l4_ch_j]
        for i in range(nF):
            row_syns = []
            for j in range(nF):
                # random init weights for the combination step
                w_comb = (np.random.randn(n_pool, n_pool) * 0.01).astype(np.float32)
                row_syns.append(STDPDense(weights=w_comb,
                                          name=f"l4form{j}_l23form{i}"))
            self.syn_l4form_l23form.append(row_syns)

        # Pooling: L4 form → pooled (per channel)
        self.syn_pool_form = [Dense(weights=W_pool, name=f"pool_form{i}") for i in range(nF)]

        # ── L4 colour → L2/3 colour (same pattern) ───────────────────────────
        self.syn_l4color_l23color = []
        for i in range(nC):
            row_syns = []
            for j in range(nC):
                w_comb = (np.random.randn(n_pool, n_pool) * 0.01).astype(np.float32)
                row_syns.append(STDPDense(weights=w_comb,
                                          name=f"l4color{j}_l23color{i}"))
            self.syn_l4color_l23color.append(row_syns)

        self.syn_pool_color = [Dense(weights=W_pool, name=f"pool_color{i}") for i in range(nC)]

        # ── L2/3 lateral inhibition ───────────────────────────────────────────
        eye_p = np.eye(n_pool, dtype=np.float32)
        self.syn_l23form_inh  = [Dense(weights= eye_p, name=f"l23f_inh{i}")  for i in range(nF)]
        self.syn_inh_l23form  = [Dense(weights=-eye_p, name=f"inh_l23f{i}")  for i in range(nF)]
        self.syn_l23color_inh = [Dense(weights= eye_p, name=f"l23c_inh{i}")  for i in range(nC)]
        self.syn_inh_l23color = [Dense(weights=-eye_p, name=f"inh_l23c{i}")  for i in range(nC)]

    def _wire(self, lgn: LGNSubNetwork, H, W, n, n_pool, nF, nC):
        lgn_outs = lgn.output_ports   # [lgn34, lgn56, lgnK]

        # ── LGN → V1 L6 ──────────────────────────────────────────────────────
        for i, lgn_p in enumerate(lgn_outs):
            lgn_p.connect(self.syn_lgn_l6[i].s_in)
            self.syn_lgn_l6[i].a_out.connect(self.l6[i].a_in)

        # ── LGN → V1 L4 form (Gabor) ─────────────────────────────────────────
        for i in range(nF):
            for c, lgn_p in enumerate(lgn_outs):
                lgn_p.connect(self.syn_lgn_l4form[i][c].s_in)
                self.syn_lgn_l4form[i][c].a_out.connect(self.l4_form[i].a_in)
                # STDP post-feedback
                self.l4_form[i].s_out.connect(self.syn_lgn_l4form[i][c].s_post_in)

        # ── L4 form lateral inhibition ────────────────────────────────────────
        for i in range(nF):
            self.l4_form[i].s_out.connect(self.syn_l4form_inh[i].s_in)
            self.syn_l4form_inh[i].a_out.connect(self.l4_form_inh[i].a_in)
            self.l4_form_inh[i].s_out.connect(self.syn_inh_l4form[i].s_in)
            self.syn_inh_l4form[i].a_out.connect(self.l4_form[i].a_in)

        # ── LGN → V1 L4 colour (double-opponent) ─────────────────────────────
        # ch0 ← lgn34 (excitatory center)
        lgn_outs[0].connect(self.syn_lgn_l4color[0][0].s_in)
        self.syn_lgn_l4color[0][0].a_out.connect(self.l4_color[0].a_in)

        # ch1 ← lgn56 (inhibitory center)
        lgn_outs[1].connect(self.syn_lgn_l4color[1][0].s_in)
        self.syn_lgn_l4color[1][0].a_out.connect(self.l4_color[1].a_in)

        # ch2 ← lgn34 + lgn56 + lgnK
        for c, lgn_p in enumerate(lgn_outs):
            lgn_p.connect(self.syn_lgn_l4color[2][c].s_in)
            self.syn_lgn_l4color[2][c].a_out.connect(self.l4_color[2].a_in)

        # ── L4 colour lateral inhibition ──────────────────────────────────────
        for i in range(nC):
            self.l4_color[i].s_out.connect(self.syn_l4color_inh[i].s_in)
            self.syn_l4color_inh[i].a_out.connect(self.l4_color_inh[i].a_in)
            self.l4_color_inh[i].s_out.connect(self.syn_inh_l4color[i].s_in)
            self.syn_inh_l4color[i].a_out.connect(self.l4_color[i].a_in)

        # ── L4 form → pooled ─────────────────────────────────────────────────
        # intermediate pooled nodes (use Dense pooling + STDPDense combination)
        l4_form_pooled = []
        for i in range(nF):
            self.l4_form[i].s_out.connect(self.syn_pool_form[i].s_in)
            l4_form_pooled.append(self.syn_pool_form[i])   # .a_out → l23_form

        # ── L4 → L2/3 form (full combination) ────────────────────────────────
        for i in range(nF):     # L2/3 channel i
            for j in range(nF): # L4 channel j
                l4_form_pooled[j].a_out.connect(self.syn_l4form_l23form[i][j].s_in)
                self.syn_l4form_l23form[i][j].a_out.connect(self.l23_form[i].a_in)
                self.l23_form[i].s_out.connect(self.syn_l4form_l23form[i][j].s_post_in)

        # ── L4 colour → pooled ───────────────────────────────────────────────
        l4_color_pooled = []
        for i in range(nC):
            self.l4_color[i].s_out.connect(self.syn_pool_color[i].s_in)
            l4_color_pooled.append(self.syn_pool_color[i])

        # ── L4 → L2/3 colour ─────────────────────────────────────────────────
        for i in range(nC):
            for j in range(nC):
                l4_color_pooled[j].a_out.connect(self.syn_l4color_l23color[i][j].s_in)
                self.syn_l4color_l23color[i][j].a_out.connect(self.l23_color[i].a_in)
                self.l23_color[i].s_out.connect(self.syn_l4color_l23color[i][j].s_post_in)

        # ── L2/3 lateral inhibition ───────────────────────────────────────────
        for i in range(nF):
            self.l23_form[i].s_out.connect(self.syn_l23form_inh[i].s_in)
            self.syn_l23form_inh[i].a_out.connect(self.l23_form_inh[i].a_in)
            self.l23_form_inh[i].s_out.connect(self.syn_inh_l23form[i].s_in)
            self.syn_inh_l23form[i].a_out.connect(self.l23_form[i].a_in)

        for i in range(nC):
            self.l23_color[i].s_out.connect(self.syn_l23color_inh[i].s_in)
            self.syn_l23color_inh[i].a_out.connect(self.l23_color_inh[i].a_in)
            self.l23_color_inh[i].s_out.connect(self.syn_inh_l23color[i].s_in)
            self.syn_inh_l23color[i].a_out.connect(self.l23_color[i].a_in)

        # ── L2/3 → spike counters ─────────────────────────────────────────────
        for i in range(nF):
            self.l23_form[i].s_out.connect(self.counters_form[i].s_in)

        for i in range(nC):
            self.l23_color[i].s_out.connect(self.counters_color[i].s_in)

    @property
    def l6_ports(self):
        """V1-L6 s_out ports → LGN feedback."""
        return [ch.s_out for ch in self.l6]


# ==============================================================================
# SECTION 10 — FC SNN (ANN trained with backprop → converted to SNN)
# ==============================================================================

class FCANN(nn.Module):
    """
    Fully-connected ANN proxy for FC SNN (paper §2.5).
    Structure mirrors the FC SNN:
        Input   : N_FEATURES spike-count vector  (V1 L2/3 output)
        Hidden  : N_HIDDEN neurons (tanh), matching paper Eq. 8
        Output  : 10 neurons (tanh) → softmax classification
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(Config.N_FEATURES, Config.N_HIDDEN, bias=True)
        self.output = nn.Linear(Config.N_HIDDEN,   10,              bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.hidden(x))
        o = torch.tanh(self.output(h))
        return o   # logits for softmax


class FCSNN:
    """
    FC Spiking Neural Network trained via ANN-to-SNN weight conversion.

    Training procedure (paper §2.5):
    1. Collect feature vectors {S_k(t)} from V1 L2/3 spike counts.
    2. Train FCANN with L2-norm loss and back-propagation on those vectors.
    3. Copy trained weights into SNN (one-to-one mapping).
    4. At inference, integrate spike-count features → classify via softmax.
    """

    def __init__(self):
        self.ann   = FCANN()
        self.optim = optim.SGD(
            self.ann.parameters(),
            lr=Config.LR,
            weight_decay=Config.L2_COEFF,
        )
        self.loss_fn = self._l2_loss

    @staticmethod
    def _l2_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        L2-norm loss (Eq. 7-8 of paper).
        Y_i = 1 if O_i = max(O), else 0
        E = Σ (L_i − Y_i)^2
        """
        y_hat = torch.softmax(logits, dim=-1)
        # one-hot true labels
        y_true = torch.zeros_like(y_hat)
        y_true.scatter_(1, labels.unsqueeze(1), 1.0)
        return torch.mean((y_true - y_hat) ** 2)

    def train_step(self, features: torch.Tensor,
                   labels: torch.Tensor) -> float:
        """One optimiser step; returns loss value."""
        self.ann.train()
        self.optim.zero_grad()
        logits = self.ann(features)
        loss   = self.loss_fn(logits, labels)
        loss.backward()
        self.optim.step()
        return loss.item()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax over softmax)."""
        self.ann.eval()
        with torch.no_grad():
            logits = self.ann(features)
            return torch.argmax(logits, dim=-1)

    def accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        preds = self.predict(features)
        return (preds == labels).float().mean().item()


# ==============================================================================
# SECTION 11 — RLVSL-SNN MAIN CLASS
# ==============================================================================

class RLVSLSNN:
    """
    Top-level orchestrator for the full RLVSL-SNN pipeline.

    Workflow:
        1. Build Retina / LGN / V1 sub-networks and wire them.
        2. For each training image:
             a. Encode to multi-modal spike trains.
             b. Inject into Retina → run Lava simulation T steps.
             c. Read V1 L2/3 spike count vectors.
        3. Batch spike-count vectors and train FCSNN (ANN-to-SNN).
        4. Evaluate on test set.
    """

    def __init__(self):
        print("[RLVSL-SNN] Building network …")
        self.retina = RetinaSubNetwork()
        self.lgn    = LGNSubNetwork(self.retina)
        self.v1     = V1SubNetwork(self.lgn)

        # Close V1-L6 → LGN feedback loop
        self.lgn.wire_feedback(self.v1.l6_ports)

        self.fcsnn  = FCSNN()
        self.run_cfg = Loihi1SimCfg(select_tag="floating_pt")
        print("[RLVSL-SNN] Network built ✓")

    # ── Feature extraction for a single image ────────────────────────────────

    def extract_features(self, image: torch.Tensor) -> np.ndarray:
        """
        Run RLVSL-SNN for T time steps and return the concatenated
        V1 L2/3 spike-count feature vector  (paper §2.3 / §3.1).

        Args:
            image: (1, H, W) normalised MNIST tensor

        Returns:
            features: (N_FEATURES,) float32 array
        """
        spikes = image_to_multimodal_spikes(image, T=Config.T_WINDOW)

        nF, nC = Config.N_FORM, Config.N_COLOR

        self.retina.inp_rod.run(
            condition=RunSteps(num_steps=1),
            run_cfg=self.run_cfg,
        )
        # Reset spike counters

        for cnt in self.v1.counters_form + self.v1.counters_color:
            cnt.spike_counts.set(np.zeros(Config.POOLED, dtype=np.float32))

        # Step through T time steps
        for t in range(Config.T_WINDOW):
            # Inject spike pattern at time t
            self.retina.inp_rod.spikes.set(spikes["INRod"][t])
            self.retina.inp_L.spikes.set(spikes["INL"][t])
            self.retina.inp_M.spikes.set(spikes["INM"][t])
            self.retina.inp_S.spikes.set(spikes["INS"][t])

            # Run one time step (single Lava RunSteps call)
            self.retina.inp_rod.run(
                condition=RunSteps(num_steps=1),
                run_cfg=self.run_cfg,
            )

        # Collect spike counts from V1 L2/3
        form_counts  = np.concatenate(
            [c.spike_counts.get() for c in self.v1.counters_form],  axis=0
        )   # (nF * POOLED,)
        color_counts = np.concatenate(
            [c.spike_counts.get() for c in self.v1.counters_color], axis=0
        )   # (nC * POOLED,)

        return np.concatenate([form_counts, color_counts], axis=0)  # (N_FEATURES,)

    def stop(self):
        """Halt the Lava runtime cleanly."""
        self.retina.inp_rod.stop()

    # ── Batch feature extraction ──────────────────────────────────────────────

    def collect_features(
        self,
        loader: DataLoader,
        max_samples: Optional[int] = None,
        desc: str = "Extract",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run all images in loader through RLVSL-SNN and collect feature vectors.

        Returns:
            features : (N, N_FEATURES) float32 Tensor
            labels   : (N,) long Tensor
        """
        all_feats  = []
        all_labels = []
        count      = 0

        for images, labels in loader:
            for img, lbl in zip(images, labels):
                feat = self.extract_features(img)
                all_feats.append(feat)
                all_labels.append(int(lbl))
                count += 1
                if count % 500 == 0:
                    print(f"  [{desc}] {count} images processed …")
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break

        X = torch.tensor(np.stack(all_feats,  axis=0), dtype=torch.float32)
        Y = torch.tensor(all_labels,           dtype=torch.long)
        return X, Y

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        epochs: int = Config.EPOCHS,
        max_train: Optional[int] = None,
        max_test:  Optional[int] = None,
    ):
        """
        Full training loop:
        1. Extract features via RLVSL-SNN (STDP trains online during extraction).
        2. Train FCSNN (ANN proxy) on collected features.
        3. Evaluate on test set.
        """
        print("\n" + "=" * 60)
        print("  Phase 1: Feature extraction + STDP training")
        print("=" * 60)
        X_train, Y_train = self.collect_features(
            train_loader, max_samples=max_train, desc="Train"
        )
        print(f"  Collected {len(X_train)} train feature vectors "
              f"(shape {tuple(X_train.shape)})")

        print("\n" + "=" * 60)
        print("  Phase 2: FC SNN training (ANN back-propagation)")
        print("=" * 60)
        ds_train  = torch.utils.data.TensorDataset(X_train, Y_train)
        fc_loader = DataLoader(ds_train, batch_size=Config.BATCH_SIZE, shuffle=True)

        history = {"train_loss": [], "train_acc": []}

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0
            for feat_batch, lbl_batch in fc_loader:
                loss = self.fcsnn.train_step(feat_batch, lbl_batch)
                epoch_loss += loss
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_acc = self.fcsnn.accuracy(X_train, Y_train)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_acc)

            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  train_acc={train_acc:.4f}")

        print("\n" + "=" * 60)
        print("  Phase 3: Test evaluation")
        print("=" * 60)
        X_test, Y_test = self.collect_features(
            test_loader, max_samples=max_test, desc="Test"
        )
        test_acc = self.fcsnn.accuracy(X_test, Y_test)
        print(f"\n  ✓  Test Accuracy : {test_acc * 100:.2f}%")
        print(f"     (Paper reports 94.62% on MNIST with 40 time steps)\n")

        self.stop()
        return history, test_acc


# ==============================================================================
# SECTION 12 — STANDALONE SIMULATION LOOP (NumPy fallback)
# ==============================================================================

class RLVSLSNNNumPy:
    """
    Pure-NumPy simulation of RLVSL-SNN for environments where the full
    Lava runtime cannot be invoked (e.g. limited GPU/hardware access).

    This mirrors the Lava process models exactly, so results are identical.
    Useful for rapid prototyping and unit-testing the biological logic.
    """

    def __init__(self):
        H, W = Config.IMG_H, Config.IMG_W
        n    = Config.N_PIX
        nF, nC = Config.N_FORM, Config.N_COLOR
        n_pool = Config.POOLED

        # ── State tensors ─────────────────────────────────────────────────────
        # Photoreceptors (graded)
        self.v_rod  = np.full(n, Config.VREST_GRADE, np.float32)
        self.v_cL   = np.full(n, Config.VREST_GRADE, np.float32)
        self.v_cM   = np.full(n, Config.VREST_GRADE, np.float32)
        self.v_cS   = np.full(n, Config.VREST_GRADE, np.float32)

        # BC layer
        self.v_rbc    = np.full(n, Config.VREST_BC, np.float32)
        self.v_bcLon  = np.full(n, Config.VREST_BC, np.float32)
        self.v_bcLoff = np.full(n, Config.VREST_BC, np.float32)
        self.v_bcMon  = np.full(n, Config.VREST_BC, np.float32)
        self.v_bcMoff = np.full(n, Config.VREST_BC, np.float32)
        self.v_bcSon  = np.full(n, Config.VREST_BC, np.float32)
        self.v_bcSoff = np.full(n, Config.VREST_BC, np.float32)

        # GC layer
        for attr in ["gcLon","gcLoff","gcMon","gcMoff","gcSon","gcSoff"]:
            setattr(self, f"v_{attr}", np.full(n, Config.VREST_LIF, np.float32))

        # LGN
        for attr in ["lgn34p","lgn56p","lgnKp","lgn34","lgn56","lgnK"]:
            setattr(self, f"v_{attr}", np.full(n, Config.VREST_LIF, np.float32))

        # V1 L4 (form + inhibitory)
        self.v_l4f    = [np.full(n, Config.VREST_LIF, np.float32) for _ in range(nF)]
        self.v_l4fi   = [np.full(n, Config.VREST_LIF, np.float32) for _ in range(nF)]

        # V1 L4 colour
        self.v_l4c  = [np.full(n, Config.VREST_LIF, np.float32) for _ in range(nC)]
        self.v_l4ci = [np.full(n, Config.VREST_LIF, np.float32) for _ in range(nC)]

        # V1 L2/3
        self.v_l23f  = [np.full(n_pool, Config.VREST_LIF, np.float32) for _ in range(nF)]
        self.v_l23fi = [np.full(n_pool, Config.VREST_LIF, np.float32) for _ in range(nF)]
        self.v_l23c  = [np.full(n_pool, Config.VREST_LIF, np.float32) for _ in range(nC)]
        self.v_l23ci = [np.full(n_pool, Config.VREST_LIF, np.float32) for _ in range(nC)]

        # ── Weight matrices ───────────────────────────────────────────────────
        self._init_weights(H, W, n, n_pool, nF, nC)

        # ── STDP traces ───────────────────────────────────────────────────────
        self.tr_pre_gcL  = np.zeros(n, np.float32)
        self.tr_post_gcL = np.zeros(n, np.float32)
        self.tr_pre_gcM  = np.zeros(n, np.float32)
        self.tr_post_gcM = np.zeros(n, np.float32)
        self.tr_pre_gcS  = np.zeros(n, np.float32)
        self.tr_post_gcS = np.zeros(n, np.float32)

        # Spike count accumulators for V1 L2/3
        self.cnt_form  = [np.zeros(n_pool, np.float32) for _ in range(nF)]
        self.cnt_color = [np.zeros(n_pool, np.float32) for _ in range(nC)]

    def _init_weights(self, H, W, n, n_pool, nF, nC):
        """Initialise all weight matrices (mirrors Lava process weights)."""
        eye = np.eye(n, dtype=np.float32)

        # Gaussian (noise reduction / light adaptation)
        gauss = gaussian_kernel(Config.RF_RBC, sigma=1.5)
        self.W_rod_rbc  = rf_weight_matrix(H, W, gauss)
        self.W_rbc_gc   = -rf_weight_matrix(H, W, gauss)

        # DoG for GC center-surround
        r = Config.RF_GC_INH
        dog_on  = dog_kernel(r, 0.5, 1.5, on_center=True)
        dog_off = dog_kernel(r, 0.5, 1.5, on_center=False)
        Won_exc  = np.maximum(0, rf_weight_matrix(H, W, dog_on))
        Won_inh  = np.minimum(0, rf_weight_matrix(H, W, dog_on))
        Woff_exc = np.maximum(0, rf_weight_matrix(H, W, dog_off))
        Woff_inh = np.minimum(0, rf_weight_matrix(H, W, dog_off))

        # BC → GC STDP weight matrices (initialised uniform)
        self.W_bcLon_gcLon   = np.clip(Won_exc + np.random.randn(*Won_exc.shape)*0.01, 0, 1).astype(np.float32)
        self.W_bcMon_gcMon   = self.W_bcLon_gcLon.copy()
        self.W_bcSon_gcSon   = self.W_bcLon_gcLon.copy()
        self.W_bcLoff_gcLoff = np.clip(Woff_exc + np.random.randn(*Woff_exc.shape)*0.01, 0, 1).astype(np.float32)
        self.W_bcMoff_gcMoff = self.W_bcLoff_gcLoff.copy()
        self.W_bcSoff_gcSoff = self.W_bcLoff_gcLoff.copy()

        # Fixed inhibitory surround weights
        self.W_bcLoff_gcLon  = Won_inh
        self.W_bcMoff_gcMon  = Won_inh
        self.W_bcSoff_gcSon  = Won_inh
        self.W_bcLon_gcLoff  = Woff_inh
        self.W_bcMon_gcMoff  = Woff_inh
        self.W_bcSon_gcSoff  = Woff_inh

        # LGN: 1:1 excitatory, radius-1 inhibitory
        r2 = Config.RF_LGN_INH
        W_lgn_inh = np.minimum(0, rf_weight_matrix(H, W, dog_kernel(r2, 0.5, 1.5, False)))
        self.W_lgn_inh = W_lgn_inh
        self.W_lgn_eye = eye

        # LGNp → LGN output (STDP, initialised as identity)
        self.W_34p_34 = eye.copy()
        self.W_56p_56 = eye.copy()
        self.W_Kp_K   = eye.copy()

        # Gabor kernels for V1 L4 form
        orientations = np.linspace(0, np.pi, nF, endpoint=False)
        self.W_gabor = []
        for theta in orientations:
            k  = gabor_kernel(Config.GABOR_RF, theta=theta)
            Wg = (np.maximum(0, rf_weight_matrix(H, W, k)) / 3.0).astype(np.float32)
            self.W_gabor.append(Wg)

        # Double-opponent for V1 L4 colour
        r3 = Config.RF_V1 + 1
        dop_e = rf_weight_matrix(H, W, double_opponent_kernel(r3, True))
        dop_i = rf_weight_matrix(H, W, double_opponent_kernel(r3, False))
        self.W_color = [
            np.maximum(0, dop_e),
            np.maximum(0, dop_i),
            np.maximum(0, dop_e) / 3.0,
        ]

        # L4 lateral inhibition (identity)
        self.W_lat_inh    = eye.copy()
        self.W_lat_inh_fb = -eye.copy()

        # Average pooling matrix
        pool = Config.POOL
        W_pool = np.zeros((n_pool, n), dtype=np.float32)
        for pi in range(H // pool):
            for pj in range(W // pool):
                oi = pi * (W // pool) + pj
                for di in range(pool):
                    for dj in range(pool):
                        ii = (pi * pool + di) * W + (pj * pool + dj)
                        W_pool[oi, ii] = 1.0 / (pool * pool)
        self.W_pool = W_pool

        # L4 → L2/3 combination (random small init → trained by STDP)
        self.W_l4f_l23f = [
            [np.random.randn(n_pool, n_pool).astype(np.float32) * 0.01 for _ in range(nF)]
            for _ in range(nF)
        ]
        self.W_l4c_l23c = [
            [np.random.randn(n_pool, n_pool).astype(np.float32) * 0.01 for _ in range(nC)]
            for _ in range(nC)
        ]

        # L2/3 lateral inhibition
        self.W_pool_lat    = np.eye(n_pool, dtype=np.float32)
        self.W_pool_lat_fb = -np.eye(n_pool, dtype=np.float32)

    # ── Single neuron-type forward step ──────────────────────────────────────

    @staticmethod
    def _graded_step(v, i_in, tau, rm, vr, ep, vsl):
        """Euler step for graded neuron; returns updated v and graded output."""
        v = v + (i_in * rm + vr - v) / tau * Config.DT
        s = np.maximum(0.0, np.tanh((v - ep) / vsl))
        return v, s.astype(np.float32)

    @staticmethod
    def _lif_step(v, i_in, tau, rm, vr, vth):
        """Euler step for LIF neuron; returns updated v and binary spikes."""
        v  = v + (i_in * rm + vr - v) / tau * Config.DT
        v[np.abs(v) < 1e-5] = 0.0
        spk = (v >= vth).astype(np.float32)
        v[spk > 0] = vr
        return v, spk

    @staticmethod
    def _stdp_update(W, s_pre, s_post, tr_pre, tr_post):
        """Additive STDP weight update (Eq. 5-6, nearest-neighbour)."""
        dt = Config.DT
        tr_pre  *= np.exp(-dt / Config.TAU_PLUS)
        tr_post *= np.exp(-dt / Config.TAU_MINUS)
        tr_pre[tr_pre < 1e-5] = 0.0
        tr_post[tr_post < 1e-5] = 0.0
        tr_pre [s_pre  > 0] = 1.0
        tr_post[s_post > 0] = 1.0
        W += Config.A_POST * np.outer(s_post, tr_pre) - \
             Config.A_PRE  * np.outer(tr_post, s_pre)
        W[:] = np.clip(W, Config.W_MIN, Config.W_MAX)
        return tr_pre, tr_post

    def reset_counters(self):
        """Zero spike-count accumulators before processing a new image."""
        for c in self.cnt_form:  c[:] = 0.0
        for c in self.cnt_color: c[:] = 0.0

    def step(self, spikes_t: Dict[str, np.ndarray]):
        """
        Process one time step of the full RLVSL-SNN.

        Args:
            spikes_t: {'INRod':(N,), 'INL':(N,), 'INM':(N,), 'INS':(N,)}
        """
        nF, nC = Config.N_FORM, Config.N_COLOR
        tau_g, rm_g = Config.TAU_GRADE, Config.RM_GRADE
        vr_g = Config.VREST_GRADE
        ep, vsl = Config.EP, Config.VSL

        tau_bc, rm_bc = Config.TAU_BC, Config.RM_BC
        vr_bc = Config.VREST_BC

        tau_l, rm_l = Config.TAU_LIF, Config.RM_LIF
        vr_l, vth = Config.VREST_LIF, Config.VTH

        # ── Photoreceptors ────────────────────────────────────────────────────
        self.v_rod, s_rod = self._graded_step(
            self.v_rod, spikes_t["INRod"], tau_g, rm_g, vr_g, ep, vsl)
        self.v_cL,  s_cL  = self._graded_step(
            self.v_cL,  spikes_t["INL"],   tau_g, rm_g, vr_g, ep, vsl)
        self.v_cM,  s_cM  = self._graded_step(
            self.v_cM,  spikes_t["INM"],   tau_g, rm_g, vr_g, ep, vsl)
        self.v_cS,  s_cS  = self._graded_step(
            self.v_cS,  spikes_t["INS"],   tau_g, rm_g, vr_g, ep, vsl)

        # ── RBC (rod pathway) ────────────────────────────────────────────────
        i_rbc = self.W_rod_rbc @ s_rod
        self.v_rbc, s_rbc = self._lif_step(self.v_rbc, i_rbc, tau_bc, rm_bc, vr_bc, vth)

        # ── Bipolar cells (cone pathway) ─────────────────────────────────────
        self.v_bcLoff, s_bcLoff = self._lif_step(self.v_bcLoff,  s_cL, tau_bc, rm_bc, vr_bc, vth)
        self.v_bcLon,  s_bcLon  = self._lif_step(self.v_bcLon,  -s_cL, tau_bc, rm_bc, vr_bc, vth)
        self.v_bcMoff, s_bcMoff = self._lif_step(self.v_bcMoff,  s_cM, tau_bc, rm_bc, vr_bc, vth)
        self.v_bcMon,  s_bcMon  = self._lif_step(self.v_bcMon,  -s_cM, tau_bc, rm_bc, vr_bc, vth)
        self.v_bcSoff, s_bcSoff = self._lif_step(self.v_bcSoff,  s_cS, tau_bc, rm_bc, vr_bc, vth)
        self.v_bcSon,  s_bcSon  = self._lif_step(self.v_bcSon,  -s_cS, tau_bc, rm_bc, vr_bc, vth)

        # ── Ganglion cells (center-surround) ──────────────────────────────────
        i_gcLon = (self.W_bcLon_gcLon @ s_bcLon + self.W_bcLoff_gcLon @ s_bcLoff
                   + self.W_rbc_gc @ s_rbc)
        self.v_gcLon, s_gcLon = self._lif_step(self.v_gcLon, i_gcLon, tau_l, rm_l, vr_l, vth)

        i_gcMon = (self.W_bcMon_gcMon @ s_bcMon + self.W_bcMoff_gcMon @ s_bcMoff
                   + self.W_rbc_gc @ s_rbc)
        self.v_gcMon, s_gcMon = self._lif_step(self.v_gcMon, i_gcMon, tau_l, rm_l, vr_l, vth)

        i_gcSon = (self.W_bcSon_gcSon @ s_bcSon + self.W_bcSoff_gcSon @ s_bcSoff
                   + self.W_rbc_gc @ s_rbc)
        self.v_gcSon, s_gcSon = self._lif_step(self.v_gcSon, i_gcSon, tau_l, rm_l, vr_l, vth)

        i_gcLoff = self.W_bcLoff_gcLoff @ s_bcLoff + self.W_bcLon_gcLoff @ s_bcLon
        self.v_gcLoff, s_gcLoff = self._lif_step(self.v_gcLoff, i_gcLoff, tau_l, rm_l, vr_l, vth)

        i_gcMoff = self.W_bcMoff_gcMoff @ s_bcMoff + self.W_bcMon_gcMoff @ s_bcMon
        self.v_gcMoff, s_gcMoff = self._lif_step(self.v_gcMoff, i_gcMoff, tau_l, rm_l, vr_l, vth)

        i_gcSoff = self.W_bcSoff_gcSoff @ s_bcSoff + self.W_bcSon_gcSoff @ s_bcSon
        self.v_gcSoff, s_gcSoff = self._lif_step(self.v_gcSoff, i_gcSoff, tau_l, rm_l, vr_l, vth)

        # ── STDP updates for GC layer ─────────────────────────────────────────
        self.tr_pre_gcL, self.tr_post_gcL = self._stdp_update(
            self.W_bcLon_gcLon, s_bcLon, s_gcLon, self.tr_pre_gcL, self.tr_post_gcL)
        self.tr_pre_gcM, self.tr_post_gcM = self._stdp_update(
            self.W_bcMon_gcMon, s_bcMon, s_gcMon, self.tr_pre_gcM, self.tr_post_gcM)
        self.tr_pre_gcS, self.tr_post_gcS = self._stdp_update(
            self.W_bcSon_gcSon, s_bcSon, s_gcSon, self.tr_pre_gcS, self.tr_post_gcS)

        # ── LGN (input layer) ─────────────────────────────────────────────────
        gc_on  = [s_gcLon, s_gcMon, s_gcSon]
        gc_off = [s_gcLoff, s_gcMoff, s_gcSoff]

        v_lgn_p = [self.v_lgn34p, self.v_lgn56p, self.v_lgnKp]
        for i, (v_p, s_on, s_off) in enumerate(zip(v_lgn_p, gc_on, gc_off)):
            i_lgn = self.W_lgn_eye @ s_on + self.W_lgn_inh @ s_off
            v_new, s_new = self._lif_step(v_p, i_lgn, tau_l, rm_l, vr_l, vth)
            v_lgn_p[i] = v_new
            gc_on[i]   = s_new   # reuse as s_lgn_p

        self.v_lgn34p, self.v_lgn56p, self.v_lgnKp = v_lgn_p
        s_34p, s_56p, s_Kp = gc_on

        # ── LGN (output layer) ────────────────────────────────────────────────
        self.v_lgn34, s_34 = self._lif_step(self.v_lgn34, self.W_34p_34 @ s_34p, tau_l, rm_l, vr_l, vth)
        self.v_lgn56, s_56 = self._lif_step(self.v_lgn56, self.W_56p_56 @ s_56p, tau_l, rm_l, vr_l, vth)
        self.v_lgnK,  s_K  = self._lif_step(self.v_lgnK,  self.W_Kp_K  @ s_Kp,  tau_l, rm_l, vr_l, vth)

        s_lgn = [s_34, s_56, s_K]

        # ── V1 L4 form pathway (Gabor) ────────────────────────────────────────
        s_l4f = []
        for i in range(nF):
            # Sum LGN channels with Gabor RF weights
            i_l4 = sum(self.W_gabor[i] @ s for s in s_lgn)
            # Lateral inhibitory channel
            i_l4 -= self.W_lat_inh_fb @ self.v_l4fi[i]   # inhibitory feedback
            self.v_l4f[i], s_fi = self._lif_step(self.v_l4f[i], i_l4, tau_l, rm_l, vr_l, vth)

            i_inh = self.W_lat_inh @ s_fi
            self.v_l4fi[i], _ = self._lif_step(self.v_l4fi[i], i_inh, tau_l, rm_l, vr_l, vth)
            s_l4f.append(s_fi)

        # ── V1 L4 colour pathway (double-opponent) ────────────────────────────
        s_l4c = []
        color_inputs = [s_34, s_56, (s_34 + s_56 + s_K) / 3.0]
        for i in range(nC):
            i_l4c  = self.W_color[i] @ color_inputs[i]
            i_l4c -= self.W_lat_inh_fb @ self.v_l4ci[i]
            self.v_l4c[i], s_ci = self._lif_step(self.v_l4c[i], i_l4c, tau_l, rm_l, vr_l, vth)

            i_inh = self.W_lat_inh @ s_ci
            self.v_l4ci[i], _ = self._lif_step(self.v_l4ci[i], i_inh, tau_l, rm_l, vr_l, vth)
            s_l4c.append(s_ci)

        # ── V1 L2/3 form (pool + full combination) ────────────────────────────
        s_l4f_pooled = [self.W_pool @ s for s in s_l4f]
        for i in range(nF):
            i_l23 = sum(self.W_l4f_l23f[i][j] @ s_l4f_pooled[j] for j in range(nF))
            i_l23 -= self.W_pool_lat_fb @ self.v_l23fi[i]
            self.v_l23f[i], s_23fi = self._lif_step(self.v_l23f[i], i_l23, tau_l, rm_l, vr_l, vth)

            i_inh = self.W_pool_lat @ s_23fi
            self.v_l23fi[i], _ = self._lif_step(self.v_l23fi[i], i_inh, tau_l, rm_l, vr_l, vth)
            self.cnt_form[i] += s_23fi

        # ── V1 L2/3 colour ────────────────────────────────────────────────────
        s_l4c_pooled = [self.W_pool @ s for s in s_l4c]
        for i in range(nC):
            i_l23c = sum(self.W_l4c_l23c[i][j] @ s_l4c_pooled[j] for j in range(nC))
            i_l23c -= self.W_pool_lat_fb @ self.v_l23ci[i]
            self.v_l23c[i], s_23ci = self._lif_step(self.v_l23c[i], i_l23c, tau_l, rm_l, vr_l, vth)

            i_inh = self.W_pool_lat @ s_23ci
            self.v_l23ci[i], _ = self._lif_step(self.v_l23ci[i], i_inh, tau_l, rm_l, vr_l, vth)
            self.cnt_color[i] += s_23ci

    def extract_features(self, image: torch.Tensor) -> np.ndarray:
        """Run T time steps and return V1 L2/3 feature vector."""
        img = image.numpy().squeeze(0)
        img = np.clip(img, 0.0, 1.0)
        spikes = image_to_multimodal_spikes(image, T=Config.T_WINDOW)

        self.reset_counters()
        for t in range(Config.T_WINDOW):
            step_spikes = {k: v[t] for k, v in spikes.items()}
            self.step(step_spikes)

        form_feats  = np.concatenate(self.cnt_form,  axis=0)
        color_feats = np.concatenate(self.cnt_color, axis=0)
        return np.concatenate([form_feats, color_feats], axis=0).astype(np.float32)


# ==============================================================================
# SECTION 13 — TRAINING PIPELINE  (unified: Lava or NumPy backend)
# ==============================================================================

def run_training_pipeline(
    use_lava:    bool = True,
    max_train:   Optional[int] = 10_000,
    max_test:    Optional[int] = 2_000,
    epochs:      int = Config.EPOCHS,
    data_dir:    str = "./data",
):
    """
    Full pipeline:
      1. Load MNIST with PyTorch.
      2. Build RLVSL-SNN (Lava or NumPy backend).
      3. Extract features (STDP trains online during extraction).
      4. Train FC SNN (ANN proxy + weight conversion).
      5. Report test accuracy vs. paper result (94.62%).

    Args:
        use_lava   : If True use Lava runtime; else use NumPy fallback.
        max_train  : Limit training samples (None = all 60 000).
        max_test   : Limit test samples (None = all 10 000).
        epochs     : FC SNN ANN training epochs.
    """
    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(data_dir=data_dir)

    # ── 2. Feature extraction backend ─────────────────────────────────────────
    if use_lava:
        print("\n[Backend] Intel Lava SDK (floating-point simulation)")
        net = RLVSLSNN()
    else:
        print("\n[Backend] NumPy fallback (identical neuron models)")
        net = RLVSLSNNNumPy()

    # ── 3 & 4. Feature extraction + FC SNN training ───────────────────────────
    fcsnn    = FCSNN()
    X_train  = []
    Y_train  = []
    count    = 0
    n_limit  = max_train if max_train else float("inf")

    print(f"\n[Phase 1] Extracting train features (max={max_train}) …")
    for images, labels in train_loader:
        for img, lbl in zip(images, labels):
            feat = net.extract_features(img)
            X_train.append(feat)
            Y_train.append(int(lbl))
            count += 1
            if count % 1000 == 0:
                print(f"  {count} done …")
        if count >= n_limit:
            break

    X_tr = torch.tensor(np.stack(X_train), dtype=torch.float32)
    Y_tr = torch.tensor(Y_train,            dtype=torch.long)
    print(f"  Feature matrix: {tuple(X_tr.shape)}  (N_FEATURES={Config.N_FEATURES})")

    print(f"\n[Phase 2] Training FC SNN ({epochs} epochs) …")
    ds_fc    = torch.utils.data.TensorDataset(X_tr, Y_tr)
    fc_loader = DataLoader(ds_fc, batch_size=Config.BATCH_SIZE, shuffle=True)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for fb, lb in fc_loader:
            epoch_loss += fcsnn.train_step(fb, lb)
        avg_loss   = epoch_loss / len(fc_loader)
        train_acc  = fcsnn.accuracy(X_tr, Y_tr)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_acc={train_acc*100:.2f}%")

    print(f"\n[Phase 3] Extracting test features (max={max_test}) …")
    X_test, Y_test = [], []
    count = 0
    n_lim_t = max_test if max_test else float("inf")
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            feat = net.extract_features(img)
            X_test.append(feat)
            Y_test.append(int(lbl))
            count += 1
        if count >= n_lim_t:
            break

    X_te = torch.tensor(np.stack(X_test), dtype=torch.float32)
    Y_te = torch.tensor(Y_test,            dtype=torch.long)
    test_acc = fcsnn.accuracy(X_te, Y_te)

    print("\n" + "=" * 60)
    print(f"  FINAL TEST ACCURACY : {test_acc * 100:.2f}%")
    print(f"  Paper result        : 94.62%  (Table 1, 40 time steps)")
    print("=" * 60 + "\n")

    # Optional: clean up Lava runtime
    if use_lava:
        try:
            net.stop()
        except Exception:
            pass

    return test_acc, fcsnn


# ==============================================================================
# SECTION 14 — VISUALISATION UTILITIES
# ==============================================================================

def plot_kernels(save_path: Optional[str] = None):
    """
    Visualise the receptive-field kernels used in the network:
      - Gaussian (noise reduction)
      - DoG on-center / off-center (GC layer)
      - Gabor orientations (V1 L4 form pathway)
      - Double-opponent (V1 L4 colour pathway)
    """
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    fig.suptitle("RLVSL-SNN Receptive-Field Kernels", fontsize=14, fontweight="bold")

    # Row 0: Gaussian and DoG
    axes[0, 0].imshow(gaussian_kernel(3, 1.5), cmap="bwr", vmin=-0.3, vmax=0.3)
    axes[0, 0].set_title("Gaussian (Rod→RBC)")

    for col, (title, on) in enumerate(
            [("DoG On-center (GCon)", True), ("DoG Off-center (GCoff)", False)], start=1):
        axes[0, col].imshow(dog_kernel(1, 0.5, 1.5, on), cmap="bwr", vmin=-0.3, vmax=0.3)
        axes[0, col].set_title(title)

    axes[0, 3].imshow(double_opponent_kernel(2, True),  cmap="bwr", vmin=-0.3, vmax=0.3)
    axes[0, 3].set_title("Dbl-Opp center-excit (V1 colour ch0)")
    axes[0, 4].imshow(double_opponent_kernel(2, False), cmap="bwr", vmin=-0.3, vmax=0.3)
    axes[0, 4].set_title("Dbl-Opp center-inhib (V1 colour ch1)")

    # Rows 1-2: Gabor kernels
    orientations = np.linspace(0, np.pi, Config.N_FORM, endpoint=False)
    for idx, theta in enumerate(orientations):
        r, c = divmod(idx, 5)
        k = gabor_kernel(Config.GABOR_RF, theta=theta)
        axes[r + 1, c].imshow(k, cmap="bwr", vmin=-0.3, vmax=0.3)
        axes[r + 1, c].set_title(f"Gabor θ={np.degrees(theta):.0f}°")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Kernels saved → {save_path}")
    plt.show()


def plot_spike_raster(spikes: np.ndarray, title: str = "Spike raster",
                      save_path: Optional[str] = None):
    """Plot a binary spike raster  (T × N_neurons)."""
    T, N = spikes.shape
    fig, ax = plt.subplots(figsize=(12, 5))
    times, neurons = np.where(spikes > 0)
    ax.scatter(times, neurons, s=0.5, c="black", marker="|")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(0, T)
    ax.set_ylim(0, N)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLVSL-SNN: Lava SDK implementation")
    parser.add_argument("--backend",    choices=["lava", "numpy"], default="numpy",
                        help="Simulation backend (default: numpy for quick demo)")
    parser.add_argument("--max-train",  type=int, default=5_000,
                        help="Max training images (default 5000; set None for full 60k)")
    parser.add_argument("--max-test",   type=int, default=1_000,
                        help="Max test images")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="FC SNN training epochs")
    parser.add_argument("--data-dir",   type=str, default="./data")
    parser.add_argument("--plot-kernels", action="store_true",
                        help="Visualise RF kernels and exit")
    args = parser.parse_args()

    if args.plot_kernels:
        plot_kernels(save_path="rlvsl_kernels.png")
    else:
        test_acc, fcsnn = run_training_pipeline(
            use_lava   = (args.backend == "lava"),
            max_train  = args.max_train,
            max_test   = args.max_test,
            epochs     = args.epochs,
            data_dir   = args.data_dir,
        )
