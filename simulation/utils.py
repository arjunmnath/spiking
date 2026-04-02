import numpy as np
import matplotlib.pyplot as plt
# utils.py
import numpy as np
from scipy.ndimage import convolve


def dog_filter(img, sigma_c=1.0, sigma_s=3.0, on_center=True):
    """
    Difference of Gaussians — models retinal ganglion / LGN receptive fields.

    sigma_c: center Gaussian radius (narrow, excitatory for ON, inhibitory for OFF)
    sigma_s: surround Gaussian radius (wide, inhibitory for ON, excitatory for OFF)
    on_center: True → ON-center/OFF-surround; False → OFF-center/ON-surround

    Returns float array same shape as img, values roughly in [-1, 1]
    """
    size = int(6 * sigma_s + 1) | 1  # odd kernel size
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2

    center = np.exp(-r2 / (2 * sigma_c ** 2)) / (2 * np.pi * sigma_c ** 2)
    surround = np.exp(-r2 / (2 * sigma_s ** 2)) / (2 * np.pi * sigma_s ** 2)

    kernel = center - surround  # ON-center: center excites, surround inhibits
    if not on_center:
        kernel = -kernel  # OFF-center: invert

    filtered = convolve(img.astype(float), kernel, mode='reflect')

    # normalise to [-1, 1] without clipping structure
    mx = np.abs(filtered).max()
    if mx > 0:
        filtered /= mx
    return filtered


def apply_lgn_filters(img, sigma_c_M=0.8, sigma_s_M=2.5,
                      sigma_c_P=0.5, sigma_s_P=1.5):
    """
    Returns six filtered maps for the six LGN streams:
      M_on, M_off   — broad receptive fields, high contrast sensitivity
      P_on, P_off   — narrow receptive fields, fine detail
      K_avg         — low-pass spatial average (koniocellular, non-specific)

    All outputs in [0, 1] (negative responses clipped — cells don't fire below rest)
    """
    img = img.astype(float)
    if img.max() > 1.0:
        img /= 255.0

    M_on_map = np.clip(dog_filter(img, sigma_c_M, sigma_s_M, on_center=True), 0, 1)
    M_off_map = np.clip(dog_filter(img, sigma_c_M, sigma_s_M, on_center=False), 0, 1)
    P_on_map = np.clip(dog_filter(img, sigma_c_P, sigma_s_P, on_center=True), 0, 1)
    P_off_map = np.clip(dog_filter(img, sigma_c_P, sigma_s_P, on_center=False), 0, 1)

    # K: smoothed luminance — diffuse, spatially non-specific
    from scipy.ndimage import gaussian_filter
    K_map = gaussian_filter(img, sigma=2.0)
    K_map = K_map / (K_map.max() + 1e-8)

    return M_on_map, M_off_map, P_on_map, P_off_map, K_map


def latency_encode(img_map, t_max=90.0, threshold=0.01):
    """
    Latency encode a single filtered map.
    High activation → early spike. Low activation → late spike.
    Pixels below threshold → no spike (cell doesn't respond).
    """
    flat = img_map.flatten().astype(float)
    mask = flat > threshold
    if mask.sum() == 0:
        return np.array([], dtype=int), np.array([])

    indices = np.where(mask)[0].astype(int)
    times = (1.0 - flat[mask]) * t_max
    times = np.clip(times, 0.5, t_max - 0.5)

    order = np.argsort(times)
    return indices[order], times[order]

def rate_encode(img, max_rate=100.0, eps=1e-8):
    img = img.astype(np.float32)
    img_flat = img.flatten()
    img_norm = img_flat / (img_flat.max() + eps)
    rates = img_norm * max_rate
    return rates

def plot_grid(X, y, n=16):
    plt.figure(figsize=(6, 6))

    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_latency_spikes(indices, times, title="Latency Spike Train"):
    plt.figure(figsize=(8, 4))

    plt.scatter(times, indices, s=8)

    plt.xlabel("Time")
    plt.ylabel("Neuron Index")
    plt.title(title)
    plt.gca().invert_yaxis()  # optional: more “neuro-style”

    plt.tight_layout()
    plt.show()


def plot_latency_spikes_2d(indices, times, shape=(16, 16)):
    plt.figure(figsize=(6, 6))

    y = indices // shape[1]
    x = indices % shape[1]

    plt.scatter(x, y, c=times, s=20)
    plt.gca().invert_yaxis()
    plt.colorbar(label="Spike Time")

    plt.title("Spatial Spike Map (colored by time)")
    plt.tight_layout()
    plt.show()

