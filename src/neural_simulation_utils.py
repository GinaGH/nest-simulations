# neural_simulation_utils.py
#
# Contains the functions used for Spiking Neural Network Simulation
# Recurrent excitatory/inhibitory network simulation
# using Poisson as structured input
#
# Helper utilities for
# NEST (Neural Simulation Technology) implementation
# to generate an Asynchronous Irregular (AI) regime
# using Brunel network with delta synapses where
# the spike trains from the individual neurons approximate
# a Poisson process

from pathlib import Path
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
import scipy.special as sp
import json
import time
import nest

#
# --- note --- this list contains some of the parameters.
#              For a full list, see config file.
#
# Network parameters for the simulation are stored in config.json
# They include the following among other parameters:
#
# - network_seed        # network_seed to regenerate the same network and connectivity
#                       # for multiple simulations.
# - sim_seed            # seed for the simulation. Choose a random ones for batch runs.
# - nu_ext_over_nu_thr
# - num_neurons
# - dt                  # simulation resolution
# - simtime             # simulation time (in ms)
# - delay               # synaptic delay
# - num_samples
# - t0
# - g_values            # preselected set of g values
# - N_rec               # number of neurons to record during simulation
# - g                   # inhibitory strength multiplier
                        # defines the ratio of inhibitory synaptic strength
                        # relative to extitatory synaptic strength
# - eta
# - epsilon             # connection probability
# - tau_syn             # simulates different receptor types on a single neuron
# - tau_m               # membrane time constant of the neuron
                        # smaller tau_m means neuron "forgets" past inputs much faster
# - C_m                 # membrane capacitance (c_mem)
# - t_ref               # refractory period duration
# - V_m                 # membrane potential (-70.0 mV is default for NEST)
# - V_reset             # reset voltage
# - V_th                # spike threshold in mV
# - E_L                 # rest potential
# - J                   # synaptic strength (weight)
# These can be overwritten if needed.
# In addition to these variables, there are some computed ones also.
# Computed parameters:
# - N_I = round(gamma * N_E)     # inhibitory neuron population is
                                 # computed as a proportion of excitatory ones
# - N_neurons = N_E + N_I        # total neuron population
# - C_E = int(epsilon * N_E)     # excitatory indegree
# - C_I = int(epsilon * N_I)     # inhibitory indegree
# - C_ext = C_E                  # external connections for Poisson input


# create a class for the configuration parameters which will
# be used to load the data from json file
class SimulationConfig:

    def __init__(self, json_path):

        # load the initial config parameters
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        self.__dict__.update(config_data)

        # compute the derived values
        self.V_th = self.theta

        # neuron population sizes based on network order
        self.NE = self.order * self.excitatory_order_multiplier
        self.NI = self.order * self.inhibitory_order_multiplier
        self.gamma = self.NI / self.NE

        print(f""""Network Order  {self.order}
            Inhibitory Neurons: {self.NI}
            Excitatory (Neurons: {self.NE}\n""")

        self.N_neurons = self.NE + self.NI      # total neuron population (N_neurons)
        self.CE = int(self.epsilon * self.NE)   # excitatory indegree
        self.CI = int(self.epsilon * self.NI)   # inhibitory indegree
        self.C_tot = int(self.CE + self.CI)     # total number of synapses per neuron
        self.C_ext = self.CE                    # external connections for Poisson input

        # the network_seed: network creation and connectivity
        # the record_seed: set of neurons to record activity of
        # the sim_seed: the seed used for simulation runs
        # to create a sequence that is deterministic, yet with nonlinear spacing
        # I use seeds based on the powers of the Golden Ratio (phi)
        #
        # phi = (1 + math.sqrt(5))/2
        seeds = generate_phi_seeds(base_power=self.phi_base_power)
        self.network_seed = seeds["network_seed"] # int(phi**10 * 1e6)
        self.record_seed  = seeds["record_seed"]  # int(phi**11 * 1e6)

        # since excitatory and inhibitory network sizes are already known
        # I create the random number of neurons to record for each
        record_rng = np.random.default_rng(self.record_seed)

        self.ex_record_indices = sorted(
            record_rng.choice(self.NE, size=self.N_rec, replace=False).tolist()
        )

        self.in_record_indices = sorted(
            record_rng.choice(self.NI, size=self.N_rec, replace=False).tolist()
        )

    def to_dict(self):
            return self.__dict__


# generate random seeds for network
# note: these are currently hardcoded in the config file
def generate_phi_seeds(base_power=10, scale=1_000_000):
    """
    Generate deterministic phi-based seeds.
    ex: int(phi**12 * 1_000_000)
    Note: these are currently hardcoded in the config file

    Returns:
        network_seed:
        record_seed
        sim_seed,
    }
    """
    import math

    phi = (1 + math.sqrt(5)) / 2

    return{
        "network_seed": int(phi**base_power * scale),
        "record_seed":  int(phi**(base_power + 1) * scale),
        "sim_seed": int(phi**(base_power + 2) * scale),
    }


# generate random seeds for network or another variation (Golden Spiral style)
# using fractional parts bounded in the full RNG range
# pseudo-random looking seeds, yet deterministic
def generate_phi_seeds_fractional(n=3, scale=2**31 - 1):
    """
    Generate deterministic phi-based seeds, using fractional part.

    Returns:
        network_seed
        record_seed
        connect_seed - combined with network_seed for now.
    """
    import math

    phi = (1 + math.sqrt(5)) / 2

    seeds = []
    for k in range(10, 10 + n):
        val = (phi**k) % 1   # fractional part
        seeds.append(int(val * scale))

    return {
        "network_seed": seeds[0],
        "record_seed":  seeds[1],
        "sim_seed": seeds[2],
    }


# the current run parameters which are stored in config
# will be written to a new json file in the outputs directory
def save_run_parameters(config, path):
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=4)


# LambertWm1 for alpha synapses - from Brunel network example
def LambertWm1(x):
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


# ComputePSPnorm for alpha synapses - from Brunel network example
def ComputePSPnorm(tauMem, CMem, tauSyn):
    """
    Compute normalization factor to convert a desired PSP amplitude (mV)
    into the correct synaptic weight for iaf_psc_alpha neurons in NEST.
    """
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem

    # time of maximum, peak PSP
    # -----  t_max = tau_syn * np.log(a) / (a - 1.0)
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return (
        np.exp(1.0)
        / (tauSyn * CMem * b)
        * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b - t_max * np.exp(-t_max / tauSyn))
    )


# spike recorder
def recorder_to_spike_trains(spike_recorder, nodes):
    events = spike_recorder.get("events")
    senders = events["senders"]
    times = events["times"]

    node_ids = nodes.tolist()
    id_to_index = {gid: i for i, gid in enumerate(node_ids)}

    spike_trains = [[] for _ in node_ids]

    for sender, spike_time in zip(senders, times):
        idx = id_to_index[sender]
        spike_trains[idx].append(spike_time)

    return spike_trains


# spike recorder
# start recording spike trains after t0 time has passed
# from the start of the simulation
# note: for now, I record everything and remove the warm up at plot-time
def recorder_to_spike_trains_after_t0(spike_recorder, nodes, t0):
    events = spike_recorder.get("events")
    senders = events["senders"]
    times = events["times"]

    node_ids = nodes.tolist()
    id_to_index = {gid: i for i, gid in enumerate(node_ids)}

    spike_trains = [[] for _ in node_ids]

    for sender, spike_time in zip(senders, times):
        if spike_time >= t0:
            idx = id_to_index[sender]
            spike_trains[idx].append(float(spike_time))

    return spike_trains


# connect_fixed_indegree_explicit
# --- note: not currently being used
def connect_fixed_indegree_explicit(
    pre_nodes,
    post_nodes,
    indegree,
    weight,
    delay,
    rng,
    allow_autapses=False,  #True allows a neuron to connect to itself
):
    """
    Explicit fixed-indegree connectivity.

    Each postsynaptic neuron receives `indegree` randomly chosen
    presynaptic neurons.

    Connectivity is controlled by the supplied NumPy rng.
    """

    pre_ids = pre_nodes.tolist()
    post_ids = post_nodes.tolist()

    n_pre = len(pre_ids)


    if indegree > n_pre:
        raise ValueError(
            f"indegree={indegree} is larger than number of presynaptic nodes={n_pre}"
        )

    same_population = set(pre_ids) == set(post_ids)

    for post_idx, post_gid in enumerate(post_ids):

        possible_indices = np.arange(n_pre)

        if (not allow_autapses) and same_population:
            possible_indices = possible_indices[possible_indices != post_idx]

        chosen_indices = rng.choice(
            possible_indices,
            size=indegree,
            replace=False,
        )

        chosen_pre_ids = sorted (
            int(pre_ids[i])
            for i in chosen_indices
        )

        # Random Balanced Network (delta synapses) notes since the Poisson generator
        # is connected to all neurons in the population, the default rule #all_to_all
        # of Connect is used.
        nest.Connect(
            chosen_pre_ids,
            [int(post_gid)],
            conn_spec={"rule": "all_to_all"},
            syn_spec={
                "weight": float(weight),
                "delay": float(delay),
            },
        )


# procedure to compute the CVs across neurons
# CV close to 1 is Poisson-like
def compute_cv_per_neuron(spike_trains, min_spikes=6):
    cvs = []

    for neuron_idx, spike_times in enumerate(spike_trains):
        spike_times = np.array(spike_times)

        if len(spike_times) >= min_spikes:
            isis = np.diff(spike_times)
            cv = np.std(isis, ddof=1) / np.mean(isis)

            cvs.append({
                "neuron": neuron_idx,
                "num_spikes": len(spike_times),
                "cv": cv,
            })

    return cvs


# procedure to simulate a recurrent network
# Alpha/Delta synapse types can be selected
# default to delta synapse and recording N_rec neurons
#
def sim_recurrent_ei_network(
    config,
    output_dir,
    prefix_file,
    number_of_runs=1, # default to one run
):
    """
    Recurrent E/I network with fixed network structure and variable trial seed.

    Uses:
        config.network_seed  -> fixed connectivity + fixed recorded neurons
        config.seed          -> trial randomness / Poisson drive

    Records only N_rec excitatory and N_rec inhibitory neurons.
    Saves only spikes occurring after config.t0.
    """

    if config.synapse_type not in {"alpha", "delta"}:
        raise ValueError("synapse_type must be 'alpha' or 'delta'")

    run_time_start = time.time() # will be used to time the full simulation (multiple runs)

    # run the simulation number_of_runs times...
    # rebuild the same network using the network_seed
    # and run a different simulation by changing the sim_seed

    # get all the simulation seeds first, and save to a list
    # to be used one per simulation run
    master_rng = np.random.default_rng(config.sim_seed)

    sim_seeds = master_rng.integers(
        low=1,
        high=2**31 - 1,
        size=number_of_runs,
        dtype=np.int64
    ).tolist()


    # --- simulate the network "number_of_runs" times
    # --- generate the network based on network_seed to ensure
    # --- the structure and connectivity remain the same across runs
    # ---
    # --- when simulating the network use the corresponding one from
    # --- the already populated list of seeds
    for run_number in range(number_of_runs):
        # time the building of the network
        startbuild = time.time()

        ### Reset kernel and build network
        nest.ResetKernel()
        nest.resolution = config.dt
        nest.rng_seed = config.network_seed # to ensure I get the same network/connectivity
        print(f"Network seed is: {config.network_seed}\n")

        # start building the prefix for all files associated with this run
        # I have the network_seed - which will be used to generate and connect the network
        RUN_PREFIX = f"snn_network_{config.network_seed}"
        # RUN_PREFIX = f"snn_network_{config.network_seed}_{time.strftime('%Y%m%d_%H%M%S')}"


        # Choose neuron model
        if config.synapse_type == "alpha":
            neuron_model = "iaf_psc_alpha"
            neuron_params = {
                "C_m": config.C_m,
                "tau_m": config.tau_m,
                "tau_syn_ex": config.tau_syn,
                "tau_syn_in": config.tau_syn,
                "t_ref": config.t_ref,
                "E_L": config.E_L,
                "V_reset": config.V_reset,
                "V_m": config.V_m,
                "V_th": config.V_th,
            }

            J_unit = ComputePSPnorm(config.tau_m, config.C_m, config.tau_syn)
            w_E = config.J / J_unit
            w_I = -config.g * w_E

        else:
            neuron_model = "iaf_psc_delta"
            neuron_params = {
                "C_m": config.C_m,
                "tau_m": config.tau_m,
                "t_ref": config.t_ref,
                "E_L": config.E_L,
                "V_reset": config.V_reset,
                "V_m": config.V_m,
                "V_th": config.V_th,
            }

            w_E = config.J
            w_I = -config.g * w_E


        print(f"Using neuron model: {neuron_model}")
        print(f"Excitatory weight: {w_E}")
        print(f"Inhibitory weight: {w_I}")

        # set other parameters
        nest.print_time = True
        nest.overwrite_files = True

        print("Building network\n")


        # Create neurons

        # E: excitatory neuron population
        # I: inhibitory neuron population

        nodes_ex = nest.Create(neuron_model, config.NE, params=neuron_params)
        nodes_in = nest.Create(neuron_model, config.NI, params=neuron_params)


        # network_seed controls the structure to ensure I have a
        # fixed recorded neuron set with the same connectivity
        # and recorded neuron selection across various simulation runs
        #
        # record_rng = same set of recorded neurons
        # network_rng = same connectivity for the network
        #
        # separate them so the state of selection doesn't affect connectivity
        # RNG state does not shift after selection
        # so I can use the SAME network and connectivity
        # and change the number of neurons recorded

        #record_rng = np.random.default_rng(config.record_seed)
        #connect_rng =np.random.default_rng(config.connect_seed)


        # External Poisson drive

        nu_th = config.theta / (config.J * config.CE * config.tau_m)
        nu_ext = config.eta * nu_th
        p_rate = 1000.0 * nu_ext * config.C_ext

        # store the variables in config
        config.run_Poisson_p_rate = p_rate
        config.run_nu_th = nu_th
        config.run_nu_ext = nu_ext

        print(f"Creating Poisson generator with p_rate: {p_rate} nu_th: {nu_th} nu_ext: {nu_ext}\n")
        print(f"C_ext: {config.C_ext} \n")
        noise = nest.Create("poisson_generator",params={"rate": p_rate})


        # connect to ex and in nodes
        nest.Connect(noise, nodes_ex, syn_spec={"weight": w_E, "delay": config.delay})
        nest.Connect(noise, nodes_in, syn_spec={"weight": w_E, "delay": config.delay}) # same as Brunel code

        E_recorded = nodes_ex[config.ex_record_indices]
        I_recorded = nodes_in[config.in_record_indices]


        # Recorders
        # note: label = ... is only important if recording to ascii
        #                   as it becomes part of the filename
        #                   I'll leave it in for now, for easy switching
        #                   to recording to "ascii" instead of "memory"

        espikes = nest.Create("spike_recorder", params = {"label" : "brunel-py-ex", "record_to": "memory"})
        ispikes = nest.Create("spike_recorder", params = {"label" : "brunel-py-in", "record_to": "memory"})

        # define a synapse type using CopyModel and overwriting some default parameters
        nest.CopyModel("static_synapse", "excitatory", {"weight": w_E, "delay": config.delay})
        nest.CopyModel("static_synapse", "inhibitory", {"weight": w_I, "delay": config.delay})

        print("Connecting spike recorder devices to neuron populations...")
        nest.Connect(E_recorded, espikes, syn_spec={"weight": w_E, "delay": config.delay})
        nest.Connect(I_recorded, ispikes, syn_spec={"weight": w_E, "delay": config.delay})


        # Connect excitatory population to all neurons

        print("Connecting network: Excitatory Connections...\n")
        conn_params_ex = {"rule": "fixed_indegree", "indegree" : config.CE}
        nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")


        # Connect inhibitory population to all neurons

        print("Connecting network: Inhibitory Connections...\n")
        conn_params_in = {"rule": "fixed_indegree", "indegree" : config.CI}
        nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

        # to endbuild time to later report time elapsed in building the network
        endbuild = time.time()


        # Simulate the network
        # this is where the current_sim_seed changes for the run

        print("Simulating network....\n")

        # keep updating the config because the seed is a part of the file name for the current run
        config.sim_seed = sim_seeds[run_number]
        nest.rng_seed = config.sim_seed # set the NEST trial seed (a new one for each simulation run)
        nest.Simulate(config.simtime)
        endsimulate = time.time()


        # Save both of the spike trains in their entirety
        # Note: the plotting/analysis functions will
        # disregard t0 time  (the warmup time)

        events_ex = espikes.n_events
        events_in = ispikes.n_events

        excitatory_spike_trains = recorder_to_spike_trains(espikes, E_recorded)
        inhibitory_spike_trains = recorder_to_spike_trains(ispikes, I_recorded)

        # these will be used for CV analysis
        excitatory_spike_trains_post_t0 = recorder_to_spike_trains_after_t0(espikes, E_recorded, config.t0)
        inhibitory_spike_trains_post_t0 = recorder_to_spike_trains_after_t0(ispikes, I_recorded, config.t0)

        # print the lengths to terminal to check....
        print(f"Excitatory spike train length {len(excitatory_spike_trains)}, should be 10\n")
        print("Excitatory spike train with sorted times: ")
        print(excitatory_spike_trains[0][:10])  # times should be sorted
        print(f"\nInhibitory spike train length {len(inhibitory_spike_trains)}, should be 10\n")

        # Save the outputs for this run
        # the file names contain the run #, network and sim seeds
        # for easy replication if needed and also
        # to group all related files for the run

        RUN_PREFIX = f"snn_network_{config.network_seed}_run_{run_number:03d}_sim_{config.sim_seed}_{time.strftime('%Y%m%d_%H%M%S')}"

        ex_path = output_dir / f"{RUN_PREFIX}_excitatory_spikes.json"
        in_path = output_dir / f"{RUN_PREFIX}_inhibitory_spikes.json"
        idx_path = output_dir / f"{RUN_PREFIX}_recorded_indices.json"
        raster_path = output_dir / f"{RUN_PREFIX}_ex_raster.png"


        with open(ex_path, "w") as f:
            json.dump(excitatory_spike_trains, f)

        with open(in_path, "w") as f:
            json.dump(inhibitory_spike_trains, f)

        with open(idx_path, "w") as f:
            json.dump(
                {
                    "network_seed": config.network_seed,
                    "simulation_record_seed": config.record_seed,
                    "simulation_run_seed": config.sim_seed,
                    "synapse_type": config.synapse_type,
                    "t0": config.t0,
                    "excitatory_indices": config.ex_record_indices,
                    "inhibitory_indices": config.in_record_indices,
                },
                f,
                indent=4,
            )

        print(f"Saved excitatory spikes to: {ex_path}")
        print(f"Saved inhibitory spikes to: {in_path}")
        print(f"Saved recorded indices to: {idx_path}")

        # compute the network firing rates, build times etc.
        # store a copy in the config to record with run data

        # build_time = endbuild - startbuild
        config.run_build_time = endbuild - startbuild

        #sim_time = endsimulate - endbuild
        config.run_simulation_time = endsimulate - endbuild

        # rate_ex = events_ex / config.simtime * 1000.0 / config.N_rec
        config.run_excitatory_rate = events_ex / config.simtime * 1000.0 / config.N_rec

        # rate_in = events_in / config.simtime * 1000.0 / config.N_rec
        config.run_inhibitory_rate = events_in / config.simtime * 1000.0 / config.N_rec

        config.run_num_ex_synapses = nest.GetDefaults("excitatory")["num_connections"]
        config.run_num_in_synapses = nest.GetDefaults("inhibitory")["num_connections"]
        config.run_tot_num_synapses = config.run_num_ex_synapses + config.run_num_in_synapses

        # print the network properties, firing rates and build times
        print("Brunel network simulation (Python)")
        print(f"Number of neurons : {config.N_neurons}")
        print(f"Number of synapses: {config.run_tot_num_synapses}")
        print(f"       Excitatory : {config.run_num_ex_synapses}")
        print(f"       Inhibitory : {config.run_num_in_synapses}")
        print(f"Excitatory rate   : {config.run_excitatory_rate:.2f} spks/s")
        print(f"Inhibitory rate   : {config.run_inhibitory_rate:.2f} spks/s")
        print(f"Building time     : {config.run_build_time:.2f} s")
        print(f"Simulation time   : {config.run_simulation_time:.2f} s")


        # Look at CV averages across neurons as behaviour one might vary
        cvs = compute_cv_per_neuron(excitatory_spike_trains_post_t0)

        for row in cvs:
            print(f"Neuron {row['neuron']}: spikes={row['num_spikes']}, CV={row['cv']:.2f}")

        config.run_mean_CV = np.mean([row["cv"] for row in cvs])
        config.run_median_CV = np.median([row["cv"] for row in cvs])

        print(f"Mean CV: {config.run_mean_CV:.2f}")
        print(f"Median CV: {config.run_median_CV:.2f}")
        print(f"Post-t0 spike counts: {[len(s) for s in excitatory_spike_trains_post_t0]}")
        print(f"P-rate for Poisson Generator: {config.run_Poisson_p_rate:.2f}")

        # Save all the parameters for this run
        print("Saving parameters for this run.\n")
        params_path = output_dir/ f"{RUN_PREFIX}_params.json"
        save_run_parameters(config, params_path)

        # print samples to terminal to check at runtime
        print("Sample excitatory Poisson spike train:")
        print(excitatory_spike_trains[0][:10])

        print("\nSample inhibitory Poisson spike train:")
        print(inhibitory_spike_trains[0][:10])

        # Append the prefix of this run to the list of runs to be analyzed later
        print(f"\nAppending prefix {RUN_PREFIX}to {prefix_file}\n")

        with open(output_dir / prefix_file, "a") as f:
                f.write(f"{RUN_PREFIX}\n")


    run_time_end = time.time()
    total_time_elapsed = run_time_end - run_time_start
    print(f"Completed {run_number + 1} simulations in {total_time_elapsed} time.\n")
    return
