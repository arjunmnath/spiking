"""
Neuromorphic Spatio-Temporal Pattern Recognizer
Backend: PyNN with Brian2 (Apple Silicon M2 compatible)
"""
import numpy as np
import matplotlib.pyplot as plt

if not hasattr(np, 'in1d'):
    np.in1d = lambda ar1, ar2, assume_unique=False, invert=False: np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)
BACKEND = 'brian2'

try:
    if BACKEND == 'brian2':
        import pyNN.brian2 as sim
    elif BACKEND == 'nest':
        import pyNN.nest as sim
    elif BACKEND == 'spiNNaker':
        import spynnaker8 as sim
except ImportError as e:
    print(f"Error importing PyNN backend '{BACKEND}'.")
    print("Please install requirements by running: pip install pyNN brian2 matplotlib")
    raise e

def run_simulation():
    dt = 0.1
    sim_duration = 400.0 
    sim.setup(timestep=dt, min_delay=1.0, max_delay=100.0)

    lif_params = {
        'cm': 0.25, 'tau_m': 20.0, 'tau_refrac': 2.0,
        'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
        'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -50.0
    }

    n_inputs = 10
    n_hidden = 5
    n_inhib = 2

    pattern_times = []
    for i in range(n_inputs):
        pattern_times.append([float(50.0 + i * 10.0), float(250.0 + i * 10.0)])
    
    input_pop = sim.Population(n_inputs, sim.SpikeSourceArray(spike_times=pattern_times), label="Sensor Array")
    noise_pop = sim.Population(n_inputs, sim.SpikeSourcePoisson(rate=15.0), label="Noise Source")
    hidden_pop = sim.Population(n_hidden, sim.IF_cond_exp(**lif_params), label="Feature Detectors")
    inhib_pop = sim.Population(n_inhib, sim.IF_cond_exp(**lif_params), label="WTA Controller")

    sim.Projection(noise_pop, hidden_pop, sim.FixedProbabilityConnector(0.5), 
                   sim.StaticSynapse(weight=0.005, delay=1.0))

    weights_in2hid = []
    for i in range(n_inputs):
        for j in range(n_hidden):
            if j == 0:
                delay = max(1.0, (n_inputs - i) * 10.0) 
                weight = 0.045
            else:
                delay = np.random.uniform(1.0, 50.0)
                weight = 0.02
            weights_in2hid.append((i, j, weight, delay))

    sim.Projection(input_pop, hidden_pop, sim.FromListConnector(weights_in2hid), receptor_type='excitatory')
    sim.Projection(hidden_pop, inhib_pop, sim.AllToAllConnector(), 
                   sim.StaticSynapse(weight=0.05, delay=1.0), receptor_type='excitatory')
    sim.Projection(inhib_pop, hidden_pop, sim.AllToAllConnector(), 
                   sim.StaticSynapse(weight=0.15, delay=1.0), receptor_type='inhibitory')

    input_pop.record(['spikes'])
    hidden_pop.record(['spikes', 'v'])
    inhib_pop.record(['spikes'])

    print(f"Running SNN via {BACKEND} backend...")
    sim.run(sim_duration)

    in_spikes = input_pop.get_data().segments[0]
    hid_data = hidden_pop.get_data().segments[0]
    sim.end()

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            pass 

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"Delay-based Spatiotemporal Coincidence Detection ({BACKEND.upper()})", fontsize=16, fontweight='bold')

    axes[0].set_title("Input Sensor Spikes (Wave Pattern + Noise)", fontsize=12)
    if len(in_spikes.spiketrains) > 0:
        for i, spiketrain in enumerate(in_spikes.spiketrains):
            times_arr = spiketrain.magnitude
            axes[0].scatter(times_arr, [i]*len(times_arr), color='#1f77b4', s=20, marker='|')
    axes[0].set_ylabel("Sensor ID")

    axes[1].set_title("Detector Layer Membrane Potentials ($V_m$)", fontsize=12)
    v_sig = hid_data.analogsignals[0]
    times = v_sig.times.magnitude

    for i in range(n_hidden):
        alpha = 1.0 if i == 0 else 0.4
        thickness = 2.0 if i == 0 else 1.0
        color = '#d62728' if i == 0 else '#7f7f7f'
        label = 'Detector 0 (Aligned Delays)' if i == 0 else ('Competing Detectors' if i == 1 else "")
        axes[1].plot(times, v_sig.magnitude[:, i], color=color, alpha=alpha, linewidth=thickness, label=label)

    if len(hid_data.spiketrains) > 0:
        for i, spiketrain in enumerate(hid_data.spiketrains):
            times_arr = spiketrain.magnitude
            color = '#d62728' if i == 0 else 'black'
            axes[1].scatter(times_arr, [0]*len(times_arr), color=color, marker='o', s=40, zorder=5)

    axes[1].axhline(lif_params['v_thresh'], color='black', linestyle='--', alpha=0.5, label='Threshold')
    axes[1].set_ylabel("Membrane Potential (mV)")
    axes[1].set_xlabel("Time (ms)", fontsize=12)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    
    output_filename = 'snn_simulation_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Simulation complete! Plot saved to '{output_filename}'")
    
if __name__ == '__main__':
    run_simulation()
