# main.py
from brian2 import *
start_scope()

import numpy as np
from cells import *
from dataset import get_numpy_cifar2_splits
from utils import latency_encode, apply_lgn_filters

# ════════════════════════════════════════════════════════════════════════════
# INPUT SPIKE GENERATOR GROUPS — one per LGN pathway
# ════════════════════════════════════════════════════════════════════════════
N = 256  # 16x16
input_M_on  = SpikeGeneratorGroup(N, [], []*ms, name='input_M_on')
input_M_off = SpikeGeneratorGroup(N, [], []*ms, name='input_M_off')
input_P_on  = SpikeGeneratorGroup(N, [], []*ms, name='input_P_on')
input_P_off = SpikeGeneratorGroup(N, [], []*ms, name='input_P_off')
input_K     = SpikeGeneratorGroup(N, [], []*ms, name='input_K')

# ════════════════════════════════════════════════════════════════════════════
# INPUT → LGN SYNAPSES
# ════════════════════════════════════════════════════════════════════════════

# M ON/OFF: 2×2 spatial pooling — broad receptive fields
m_pre, m_post = [], []
for k in range(8):
    for l in range(8):
        for dk, dj in [(0,0),(0,1),(1,0),(1,1)]:
            m_pre.append((2*k+dk)*16 + (2*l+dj))
            m_post.append(k*8+l)

S_in_Mon  = Synapses(input_M_on,  M_on,  'w:1', on_pre='v_post += w', name='S_in_Mon')
S_in_Moff = Synapses(input_M_off, M_off, 'w:1', on_pre='v_post += w', name='S_in_Moff')
S_in_Mon.connect(i=m_pre,  j=m_post); S_in_Mon.w  = 0.4
S_in_Moff.connect(i=m_pre, j=m_post); S_in_Moff.w = 0.4

# P ON/OFF: 1-to-1 — fine spatial detail
S_in_Pon  = Synapses(input_P_on,  P_on,  'w:1', on_pre='v_post += w', name='S_in_Pon')
S_in_Poff = Synapses(input_P_off, P_off, 'w:1', on_pre='v_post += w', name='S_in_Poff')
S_in_Pon.connect(j='i');  S_in_Pon.w  = 8.0
S_in_Poff.connect(j='i'); S_in_Poff.w = 8.0

# K: sparse random — diffuse luminance drive
S_in_K = Synapses(input_K, K, 'w:1', on_pre='v_post += w', name='S_in_K')
S_in_K.connect(p=0.15); S_in_K.w = 5.0

# ════════════════════════════════════════════════════════════════════════════
# INTERNEURON SYNAPSES — feedforward surround inhibition
# LGN local interneurons sharpen contrast at the neural level,
# complementing the DoG filter already applied in utils.py
# ════════════════════════════════════════════════════════════════════════════

S_in_InhM = Synapses(input_M_on, Inh_M, 'w:1', on_pre='v_post += w', name='S_in_InhM')
S_in_InhM.connect(p=0.3); S_in_InhM.w = 0.5

S_in_InhP = Synapses(input_P_on, Inh_P, 'w:1', on_pre='v_post += w', name='S_in_InhP')
S_in_InhP.connect(p=0.3); S_in_InhP.w = 0.5

S_InhM_Mon  = Synapses(Inh_M, M_on,  'w:1', on_pre='v_post -= w', name='S_InhM_Mon')
S_InhM_Moff = Synapses(Inh_M, M_off, 'w:1', on_pre='v_post -= w', name='S_InhM_Moff')
S_InhP_Pon  = Synapses(Inh_P, P_on,  'w:1', on_pre='v_post -= w', name='S_InhP_Pon')
S_InhP_Poff = Synapses(Inh_P, P_off, 'w:1', on_pre='v_post -= w', name='S_InhP_Poff')

S_InhM_Mon.connect(p=0.4);  S_InhM_Mon.w  = 0.3
S_InhM_Moff.connect(p=0.4); S_InhM_Moff.w = 0.3
S_InhP_Pon.connect(p=0.4);  S_InhP_Pon.w  = 0.2
S_InhP_Poff.connect(p=0.4); S_InhP_Poff.w = 0.2

# ════════════════════════════════════════════════════════════════════════════
# READOUT SYNAPSES: LGN → output
#
# KEY CHANGES vs previous version:
#   1. p=0.5 sparse connectivity so each output neuron sees a DIFFERENT
#      random subset of LGN neurons — breaks symmetry from the start
#   2. Small uniform init (not rand()*0.3) — prevents one neuron dominating
#   3. No STDP equations — weights updated externally via rate-coded
#      eligibility traces (see update_weights below)
# ════════════════════════════════════════════════════════════════════════════
np.random.seed(42)  # reproducible initial connectivity

def make_readout(src, name):
    S = Synapses(src, output, 'w:1', on_pre='v_post += w', name=name)
    S.connect(p=0.5)
    S.w = 'rand()*0.1 + 0.05'
    return S

S_Mon_out  = make_readout(M_on,  'S_Mon_out')
S_Moff_out = make_readout(M_off, 'S_Moff_out')
S_Pon_out  = make_readout(P_on,  'S_Pon_out')
S_Poff_out = make_readout(P_off, 'S_Poff_out')
S_K_out    = make_readout(K,     'S_K_out')

all_readout = [S_Mon_out, S_Moff_out, S_Pon_out, S_Poff_out, S_K_out]

# Pathway-specific learning rates — M fastest, K slowest
lrs = {
    'S_Mon_out':  0.02,
    'S_Moff_out': 0.02,
    'S_Pon_out':  0.015,
    'S_Poff_out': 0.015,
    'S_K_out':    0.01,
}

w_max = 1.0

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT WTA INHIBITION
# Strong mutual inhibition — the first output neuron to fire suppresses
# the other, creating a genuine winner-take-all competition
# ════════════════════════════════════════════════════════════════════════════
S_out_inh = Synapses(output, output, 'w:1', on_pre='v_post -= w',
                     name='S_out_inh')
S_out_inh.connect(condition='i != j')
S_out_inh.w = 2.0   # strong enough to push loser below threshold

# ════════════════════════════════════════════════════════════════════════════
# MONITORS
# ════════════════════════════════════════════════════════════════════════════
spike_mon      = SpikeMonitor(output,  name='spike_mon')
spike_mon_Mon  = SpikeMonitor(M_on,   name='spike_mon_Mon')
spike_mon_Moff = SpikeMonitor(M_off,  name='spike_mon_Moff')
spike_mon_Pon  = SpikeMonitor(P_on,   name='spike_mon_Pon')
spike_mon_Poff = SpikeMonitor(P_off,  name='spike_mon_Poff')
spike_mon_K    = SpikeMonitor(K,      name='spike_mon_K')

# Maps monitor name suffix → monitor object (used in update_weights)
monitors = {
    'M_on':  spike_mon_Mon,
    'M_off': spike_mon_Moff,
    'P_on':  spike_mon_Pon,
    'P_off': spike_mon_Poff,
    'K':     spike_mon_K,
}

# ════════════════════════════════════════════════════════════════════════════
# NETWORK
# ════════════════════════════════════════════════════════════════════════════
net = Network(
    # inputs
    input_M_on, input_M_off, input_P_on, input_P_off, input_K,
    # principal LGN cells
    M_on, M_off, P_on, P_off, K, output,
    # interneurons
    Inh_M, Inh_P,
    # feedforward synapses
    S_in_Mon, S_in_Moff, S_in_Pon, S_in_Poff, S_in_K,
    # inhibitory synapses
    S_in_InhM, S_in_InhP,
    S_InhM_Mon, S_InhM_Moff, S_InhP_Pon, S_InhP_Poff,
    # readout synapses
    S_Mon_out, S_Moff_out, S_Pon_out, S_Poff_out, S_K_out,
    # WTA
    S_out_inh,
    # monitors
    spike_mon, spike_mon_Mon, spike_mon_Moff,
    spike_mon_Pon, spike_mon_Poff, spike_mon_K,
)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def set_input(spike_gen, img_map, t_offset, t_max=90.0):
    idx, tms = latency_encode(img_map, t_max=t_max)
    if len(idx) == 0:
        spike_gen.set_spikes([], []*ms)
        return
    spike_gen.set_spikes(idx, tms*ms + t_offset)


def spikes_in(mon, t0, t1, n=None):
    mask = (mon.t >= t0) & (mon.t < t1)
    if n is None:
        return int(mask.sum())
    return np.bincount(np.array(mon.i)[mask], minlength=n) if mask.any() \
           else np.zeros(n, dtype=int)


# In update_weights, rename the local variable to avoid collision
def update_weights(t_start, t_end, label, pred):
    if pred == label:
        return
    correct = label
    wrong   = 1 - label

    for S in all_readout:
        src_name = S.source.name
        mon      = monitors[src_name]
        mask     = (mon.t >= t_start) & (mon.t < t_end)

        if mask.any():
            pre_counts = np.bincount(
                np.array(mon.i)[mask], minlength=S.source.N).astype(float)
        else:
            pre_counts = np.zeros(S.source.N, dtype=float)

        max_count = pre_counts.max()
        if max_count > 0:
            pre_counts /= max_count

        lr      = lrs[S.name]
        w_arr   = np.array(S.w).copy()   # <-- renamed: w_arr not w
        pre_idx = np.array(S.i[:])
        post_idx= np.array(S.j[:])

        activity = pre_counts[pre_idx]

        onto_correct = post_idx == correct
        onto_wrong   = post_idx == wrong

        w_arr[onto_correct] = np.clip(w_arr[onto_correct] + lr * activity[onto_correct], 0, w_max)
        w_arr[onto_wrong]   = np.clip(w_arr[onto_wrong]   - lr * activity[onto_wrong],   0, w_max)

        # Homeostatic weight normalization per output neuron to prevent mode collapse
        for out_n in [0, 1]:
            mask = post_idx == out_n
            w_sum = w_arr[mask].sum()
            target_sum = 0.1 * np.sum(mask) # maintain mean weight ~0.1
            if w_sum > 0:
                w_arr[mask] *= (target_sum / w_sum)

        S.w = w_arr   # assign back via attribute, not a loose 'w' variable

# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════
X_train, y_train, X_test, y_test = get_numpy_cifar2_splits()

# ════════════════════════════════════════════════════════════════════════════
# SANITY CHECK — verify all pathways fire before training
# ════════════════════════════════════════════════════════════════════════════
print("=== Sanity check ===")
maps = apply_lgn_filters(X_train[0])
M_on_map, M_off_map, P_on_map, P_off_map, K_map = maps

print(f"  Filter ranges:")
print(f"    M_on:  max={M_on_map.max():.3f}  nonzero={(M_on_map>0.01).sum()}")
print(f"    M_off: max={M_off_map.max():.3f}  nonzero={(M_off_map>0.01).sum()}")
print(f"    P_on:  max={P_on_map.max():.3f}  nonzero={(P_on_map>0.01).sum()}")
print(f"    P_off: max={P_off_map.max():.3f}  nonzero={(P_off_map>0.01).sum()}")
print(f"    K:     max={K_map.max():.3f}  nonzero={(K_map>0.01).sum()}")

t0 = net.t
set_input(input_M_on,  M_on_map,  t0)
set_input(input_M_off, M_off_map, t0)
set_input(input_P_on,  P_on_map,  t0)
set_input(input_P_off, P_off_map, t0)
set_input(input_K,     K_map,     t0)
net.run(100*ms)
t1 = net.t

print(f"  Spike counts:")
print(f"    M_on={spikes_in(spike_mon_Mon,t0,t1)}"
      f"  M_off={spikes_in(spike_mon_Moff,t0,t1)}"
      f"  P_on={spikes_in(spike_mon_Pon,t0,t1)}"
      f"  P_off={spikes_in(spike_mon_Poff,t0,t1)}"
      f"  K={spikes_in(spike_mon_K,t0,t1)}"
      f"  out={spikes_in(spike_mon,t0,t1,2)}")
print(f"  delta_t={(t1-t0)/ms:.1f} ms")
print("=== done ===\n")

# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════
X0 = X_train[y_train==0][:5]; y0 = y_train[y_train==0][:5]
X1 = X_train[y_train==1][:5]; y1 = y_train[y_train==1][:5]
X_batch = np.concatenate([X0, X1])
y_batch = np.concatenate([y0, y1])

TRIAL_MS = 100

print("Starting training\n")
for epoch in range(20):
    epoch_correct = 0

    for img, label in zip(X_batch, y_batch):
        M_on_map, M_off_map, P_on_map, P_off_map, K_map = apply_lgn_filters(img)

        t_start = net.t
        set_input(input_M_on,  M_on_map,  t_start)
        set_input(input_M_off, M_off_map, t_start)
        set_input(input_P_on,  P_on_map,  t_start)
        set_input(input_P_off, P_off_map, t_start)
        set_input(input_K,     K_map,     t_start)

        net.run(TRIAL_MS * ms)
        t_end = net.t

        counts = spikes_in(spike_mon,   t_start, t_end, n=2)
        n_Mon  = spikes_in(spike_mon_Mon,  t_start, t_end)
        n_Moff = spikes_in(spike_mon_Moff, t_start, t_end)
        n_Pon  = spikes_in(spike_mon_Pon,  t_start, t_end)
        n_Poff = spikes_in(spike_mon_Poff, t_start, t_end)
        n_K    = spikes_in(spike_mon_K,    t_start, t_end)

        # decode with tiebreak on last spike time
        if counts[0] == counts[1]:
            window = (spike_mon.t >= t_start) & (spike_mon.t < t_end)
            pred = int(np.array(spike_mon.i)[window][-1]) if window.any() \
                   else np.random.randint(2)
        else:
            pred = int(np.argmax(counts))

        correct = int(pred == label)
        epoch_correct += correct

        # rate-coded eligibility trace weight update
        update_weights(t_start, t_end, label, pred)

        print(f"  label={label} pred={pred} r={'+' if correct else '-'} "
              f"out={counts} | M+={n_Mon} M-={n_Moff} "
              f"P+={n_Pon} P-={n_Poff} K={n_K}")

    # per-epoch weight divergence diagnostic
    w_0 = np.mean([np.array(S.w)[np.array(S.j[:]) == 0].mean()
                   for S in all_readout])
    w_1 = np.mean([np.array(S.w)[np.array(S.j[:]) == 1].mean()
                   for S in all_readout])
    diff = abs(w_0 - w_1)

    print(f"Epoch {epoch:3d}  acc={epoch_correct/len(y_batch):.2f}"
          f"  w→0={w_0:.4f}  w→1={w_1:.4f}  diff={diff:.4f}\n")

    # early convergence check
    if diff > 0.05 and epoch > 10:
        print("Weights diverging — learning signal detected\n")


# After training, check per-image consistency
print("\n=== Per-image weight analysis ===")
for i, (img, label) in enumerate(zip(X_batch, y_batch)):
    M_on_map, M_off_map, P_on_map, P_off_map, K_map = apply_lgn_filters(img)

    # Mean weight toward each output for this image's active neurons
    maps_dict = {
        'M_on': M_on_map, 'M_off': M_off_map,
        'P_on': P_on_map, 'P_off': P_off_map,
        'K': K_map
    }

    score = np.zeros(2)
    for S in all_readout:
        src = S.source.name
        flat = maps_dict[src].flatten()
        active = flat > 0.01
        pre = np.array(S.i[:])
        post = np.array(S.j[:])
        w_eval = np.array(S.w)
        for n in [0, 1]:
            mask = (post == n) & active[pre]
            score[n] += w_eval[mask].sum() * flat[pre[mask]].sum()

    print(f"  img={i} label={label}  score→0={score[0]:.1f}  "
          f"score→1={score[1]:.1f}  pred={'0' if score[0] > score[1] else '1'}")

# Quick test set evaluation
print("\n=== Test set evaluation ===")
X0t = X_test[y_test==0][:20]; y0t = y_test[y_test==0][:20]
X1t = X_test[y_test==1][:20]; y1t = y_test[y_test==1][:20]
X_test_batch = np.concatenate([X0t, X1t])
y_test_batch = np.concatenate([y0t, y1t])

test_correct = 0
for img, label in zip(X_test_batch, y_test_batch):
    M_on_map, M_off_map, P_on_map, P_off_map, K_map = apply_lgn_filters(img)

    t_start = net.t
    set_input(input_M_on,  M_on_map,  t_start)
    set_input(input_M_off, M_off_map, t_start)
    set_input(input_P_on,  P_on_map,  t_start)
    set_input(input_P_off, P_off_map, t_start)
    set_input(input_K,     K_map,     t_start)
    net.run(TRIAL_MS * ms)
    t_end = net.t

    counts = spikes_in(spike_mon, t_start, t_end, n=2)
    if counts[0] == counts[1]:
        window = (spike_mon.t >= t_start) & (spike_mon.t < t_end)
        pred = int(np.array(spike_mon.i)[window][-1]) if window.any() \
               else np.random.randint(2)
    else:
        pred = int(np.argmax(counts))

    test_correct += int(pred == label)
    print(f"  label={label} pred={pred} out={counts}")

print(f"\nTest accuracy: {test_correct/len(y_test_batch):.2f} "
      f"({test_correct}/{len(y_test_batch)})")