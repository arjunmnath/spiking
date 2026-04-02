# cells.py
from brian2 import *

eqs_lif = '''
dv/dt = (-v + I) / tau : 1 (unless refractory)
I : 1
tau : second
'''

eqs_izh = '''
dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
du/dt = (a*(b*v - u)) / ms : 1
I : 1
a : 1
b : 1
c : 1
d : 1
'''

# ── Magnocellular ON/OFF (LIF, fast transient) ────────────────────────────────
M_on = NeuronGroup(64, eqs_lif, threshold='v > 1', reset='v = 0',
                   refractory=2*ms, method='euler', name='M_on')
M_on.tau = 10*ms; M_on.I = 0

M_off = NeuronGroup(64, eqs_lif, threshold='v > 1', reset='v = 0',
                    refractory=2*ms, method='euler', name='M_off')
M_off.tau = 10*ms; M_off.I = 0

# ── Parvocellular ON/OFF (Izhikevich RS, sustained) ───────────────────────────
P_on = NeuronGroup(256, eqs_izh, threshold='v >= 30',
                   reset='v = c; u += d', method='euler', name='P_on')
P_on.a=0.02; P_on.b=0.2; P_on.c=-65; P_on.d=8
P_on.v=-65;  P_on.u='b*v'; P_on.I=3.5

P_off = NeuronGroup(256, eqs_izh, threshold='v >= 30',
                    reset='v = c; u += d', method='euler', name='P_off')
P_off.a=0.02; P_off.b=0.2; P_off.c=-65; P_off.d=8
P_off.v=-65;  P_off.u='b*v'; P_off.I=3.5

# ── Koniocellular (Izhikevich bursting, slow modulatory) ──────────────────────
K = NeuronGroup(64, eqs_izh, threshold='v >= 30',
                reset='v = c; u += d', method='euler', name='K')
K.a=0.02; K.b=0.2; K.c=-50; K.d=2
K.v=-65;  K.u='b*v'; K.I=2.0

# ── Interneurons (LIF, fast inhibitory) ───────────────────────────────────────
# Perigeniculate (PGN) — feedback inhibition from cortex (not used here)
# Local interneurons — feedforward inhibition within LGN layers
Inh_M = NeuronGroup(32, eqs_lif, threshold='v > 1', reset='v = 0',
                    refractory=1*ms, method='euler', name='Inh_M')
Inh_M.tau = 3*ms; Inh_M.I = 0   # faster than principal cells

Inh_P = NeuronGroup(64, eqs_lif, threshold='v > 1', reset='v = 0',
                    refractory=1*ms, method='euler', name='Inh_P')
Inh_P.tau = 3*ms; Inh_P.I = 0

# ── Output layer ──────────────────────────────────────────────────────────────
output = NeuronGroup(2, eqs_lif, threshold='v > 1', reset='v = 0',
                     refractory=5*ms, method='euler', name='output')
output.tau = 50*ms; output.I = 0