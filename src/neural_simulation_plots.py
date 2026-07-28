# To run after the simulator data is generated to create the plots that analyze the data.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import neural_simulation_plotting_utils as nspu

# simulation runs contain a file prefix for the output files that includes the
# run sequence number, network and simulation seeds which are generated during runtime.
# these prefixes are stored in a text file so the data can be analyzed in batch
BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "outputs"
PREFIX_FILE = OUT / "_simulation_prefixes_to_analyze.txt"


# load the prefixes of files to be analyzed from the text file.
def read_prefixes(prefix_file):
    '''Read simulation filename prefixes from a text file.'''
    with open(prefix_file, "r") as f:
        prefixes = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    return prefixes


# Poisson plot comparison similar to Brian2 version in inputs [131-132]
def plot_poisson_comparison (
        ex_spikes,
        output_path,
        simtime,
        dt):
    ''' Poisson plot comparison to Brian2 in [131] plot'''

    cumulative_counts = nspu.cumulative_counts_matrix(ex_spikes, simtime=simtime, dt=dt)

    empirical_mean_ex = np.mean(cumulative_counts, axis=0)
    empirical_std_ex = np.std(cumulative_counts, axis=0, ddof=1)

    # make a time axis in milliseconds, then convert it to seconds.
    # should return something like [0, 1, 2, 3, ..., 1000]
    # with dt = 1.0 (ms) each index corresponds directly to a simulation timestep
    # index 0 -> 0 ms
    # index 1 -> 1 ms etc.
    # time_axis = np.arange(len(empirical_mean_ex)) # ms

    time_axis = np.arange(cumulative_counts.shape[1]) * dt   # ms
    time_sec = time_axis / 1000.0                     # sec

    # convert to seconds [0.0, 0.0001, 0.0002, ..., 1.0] # seconds
    # so we can compute rate in Hz (spikes per seconds)
    time_sec = time_axis / 1000.0

    # estimating firing rate from data (final cumulative spike count)
    rate_hz = empirical_mean_ex[-1] / time_sec[-1]

    # Poisson-process assumptions for comparison
    # mean count = lambda t   (lambda in HZ (spikes/sec), t in seconds
    # standard deviation = sqrt (lambda t)
    theoretical_mean = rate_hz * time_sec
    theoretical_std = np.sqrt(theoretical_mean)

    # duplicate plot of neural_simulation_main11.19.ipynb input [132]
    #
    # plot compares:
    #   - observed excitatory cumulative spike counts
    #   vs.
    #   - theoretical homogeneous Poisson spike counts
    #
    plt.figure(figsize=(10, 6))

    # plot both means --- match Brian2 colors
    # for plotting use the milliseconds to match simulation steps
    plt.plot(time_axis, theoretical_mean, label="Homogeneous Mean", color="blue")
    plt.plot(time_axis, empirical_mean_ex, label="Inhomogeneous Mean", color="red")

    # add shaded regions
    # theoretical mean +/- theoretical std
    plt.fill_between(
        time_axis,
        theoretical_mean - theoretical_std,
        theoretical_mean + theoretical_std,
        color="blue",
        alpha=0.2,
        label="Homogeneous Std Dev"
    )

    # empirical mean +/- empirical std
    plt.fill_between(
        time_axis,
        empirical_mean_ex - empirical_std_ex,
        empirical_mean_ex + empirical_std_ex,
        color="red",
        alpha=0.2,
        label="Inhomogeneous Std Dev"
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Cumulative Spike Count")
    plt.title("Excitatory Network Mean and Standard Deviation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# Batch analysis of data file
def analyze_one_simulation(prefix):
    """Analyze one simulation run given its filename prefix."""

    ex_file = OUT / f"{prefix}_excitatory_spikes.json"
    in_file = OUT / f"{prefix}_inhibitory_spikes.json"

    if not ex_file.exists():
        print(f"Missing excitatory file: {ex_file}")
        return

    if not in_file.exists():
        print(f"Missing inhibitory file: {in_file}")
        return

    # load the run parameters from the run-specific json file
    # these are all of the initial parameters,
    # along with all computed ones, as well as some initial stats
    params = nspu.load_sim_params(prefix, OUT)

    simtime = params["simtime"]
    t0 = params["t0"]
    dt = params["dt"]
    bin_dt = params["bin_dt"]

    print(f"Analyzing {prefix}")
    ex_spikes_full = nspu.load_spike_trains(ex_file)
    in_spikes_full = nspu.load_spike_trains(in_file)

    # t0 is the designated timestamp that ends the "warmup" stage
    # of the network, after which we assume the network is
    # in a steady state.
    # t0 is the time to discard from the total run time
    # ex: simtime = 1000 ms, t0=200 --> discard the first 200 ms
    ex_spikes_analysis = nspu.discard_warmup_spikes(
        ex_spikes_full,
        t0=t0,
        shift_time=True,
    )

    in_spikes_analysis = nspu.discard_warmup_spikes(
        in_spikes_full,
        t0=t0,
        shift_time=True,
    )

    sim_analysis_length = simtime - t0

    # scatter plot with different colors per line
    nspu.spike_scatter_plot(
        ex_spikes_analysis,
        in_spikes_analysis,
        OUT / f"{prefix}_nspu.spike_scatter_plot.png",
        t_start=0.0,
        t_end=sim_analysis_length,
        n_plot=25,
        bin_width=bin_dt,
        rate_ylim= None #(0, t0)
    )

    # the main plot to compare the cones
    # plot_poisson_comparison(
    #    ex_spikes_analysis,
    #    OUT / f"{prefix}_excitatory_poisson_comparison.png",
    #    simtime=sim_analysis_length,
    #    dt=bin_dt,
    #)


# process all the output files (spike trains) for the current run in the outputs directory
# the prefixes for the files are written to a .txt file
def main():
    ''' Generate the plots based on simulation data read from output files'''

    # load all the runs and batch process them for each ex and in runs ...
    # counts = cumulative_counts_tensor(all_ex_runs, simtime=sim_analysis_length, dt=dt)
    # Z = standardize_homogeneous_poisson_counts(counts, dt=dt)
    # cov = covariance_from_standardized_runs(Z)

    # Batch runs
    print("Generating plots for batch simulations...\n")
    prefixes = read_prefixes(PREFIX_FILE)

    # create the Spike Scatter Plot for the **first run** only
    # we can create more if we need it for analysis
    # this is just to give us a visual of any regularities
    # that might be present in the sample neurons' spike times.
    analyze_one_simulation(prefixes[0]) # spike raster only for the 1st run

    # the simtime, t0 and bin_dt are set across all runs,
    # so we can load them from just the first run's parameters
    run_params = nspu.load_sim_params(prefixes[0], OUT)
    simtime = run_params["simtime"]
    t0 = run_params["t0"]
    bin_dt = run_params["bin_dt"]  # IMP: bin-delta-t is different than dt for the runs
    #### dt =1 # run_params["dt"] we're setting the delta-t for bin sizes in config

    sim_analysis_length = simtime - t0

    # Load the excitatory spike trains for all runs
    # all_ex_runs shape will be
    #    [#runs, #neurons, variable length array of spike times]
    all_ex_runs = nspu.load_allruns_spike_trains(
        prefixes,
        OUT,
        population = "excitatory",
    )

    # load the inhibitory spike trains for all runs
    all_in_runs = nspu.load_allruns_spike_trains(
        prefixes,
        OUT,
        population = "inhibitory",
    )

    # Discard the warmup period from the spike trains
    # all_ex_runs shape will be the same with spike times before t0 discarded
    #    [#runs, #neurons, variable length array of spike times]

    # discard the warmup period from all excitatory trains
    all_ex_runs = nspu.discard_warmup_all_runs(
        all_ex_runs,
        t0 = t0,
        shift_time = True, # new time indices will start from 0
    )

    # discard the warmup period from all inhibitory trains
    all_in_runs = nspu.discard_warmup_all_runs(
        all_in_runs,
        t0 = t0,
        shift_time = True,
    )

    # Convert the spike trains to binary
    # binary_ex shape will be
    #   [#runs, #neurons, #bins = sim_analysis_length/bin_dt array of 0/1s]
    # ex: [100,50,800]

    # convert all of the excitatory spike times to binary
    binary_ex = nspu.spike_runs_to_binary_tensor(
        all_ex_runs,                    # all runs spike trains after warmup
        simtime=sim_analysis_length,    # analysis time after warmup
        dt=bin_dt,                      # bin resolution: delta-time
    )

    # convert all of the inhibitory spike times to binary
    binary_in = nspu.spike_runs_to_binary_tensor(
        all_in_runs,
        simtime=sim_analysis_length,
        dt=bin_dt,
    )

    # Apply the counting process to the excitatory and inhibitory spike trains
    # count_ex shape [#runs, #neurons, #bins] with cumulative counts instead of binary
    count_ex = nspu.binary_tensor_to_counting_process(binary_ex)
    count_in = nspu.binary_tensor_to_counting_process(binary_in)


    # For quick checks at runtime: print the array sizes, sample entries etc.
    print("binary_ex shape:", binary_ex.shape)
    print("count_ex shape:", count_ex.shape)
    print(binary_ex[0,0,:20])

    print("binary_in shape:", binary_in.shape)
    print("count_in shape:", count_in.shape)
    print(count_ex[0,0,:20])

    # check that the binary spike total = final cumulative count
    print(np.sum(binary_ex[0,0,:]))
    print(count_ex[0,0,-1])

    # to test: print the first 10 spike bin indices for one neuron (time bins, and in seconds)
    # for...  run 0, neuron 0, all time bins [0,0,:]
    spike_bins = np.where(binary_ex[0,0,:] == 1)[0]
    print(spike_bins[:10])
    print(spike_bins[:10] * bin_dt)

    #  Standardize the cumulative counts three different ways
    #    1. Homogeneous Assumption (constant rate): using theoretical mean and standard deviation
    #    2. Inhomogeneous (rate changes with time): using empirical mean and sqrt of the empirical mean
    #    3. Assume: not Poisson, standardize using empirical mean and empirical std

    #region First Standardization: Homogeneous Poisson Assumption

    # First: HOMOGENEOUS POISSON ASSUMPTION (theoretical mean, std)
    #   Run the standardization using theoretical mean and standard deviation
    #
    #   count_ex_standardized_XXX shape [#runs, #neurons, #bins]
    #           where each time bin data is the Z score based on Poisson assumption
    Z_count_ex_standardized_hom = nspu.standardize_counts(count_ex, mode="Homogeneous")
    Z_count_in_standardized_hom = nspu.standardize_counts(count_in, mode="Homogeneous")

    # print shapes, expect: (runs, neurons, time-bins)
    print(f"Shape of Z score matrix of covariances under Homogeneous Poisson assumption - ex: {Z_count_ex_standardized_hom.shape}")
    print(f"Shape of Z score matrix of covariances under Homogeneous Poisson assumption - in: {Z_count_in_standardized_hom.shape}")


    # Reshape standardized counts for covariance matrix by
    #    combining spike data for neurons across all_observations
    #
    # input [#runs, #neurons, #bins]
    # which returns:
    #   [#neurons, #runs x #bins (i.e. all observations)]
    #    ex: [50, 8000 = 100 runs x 800 bins]
    reshaped_Z_count_ex_hom = nspu.reshape_standardized_counts_for_covariance(Z_count_ex_standardized_hom)
    reshaped_Z_count_in_hom = nspu.reshape_standardized_counts_for_covariance(Z_count_in_standardized_hom)

    print(reshaped_Z_count_ex_hom.shape)
    print(reshaped_Z_count_in_hom.shape)  # expect: (neurons, runs x bins = total_observations)


    # Generate the covariance matrix of Z scores for the standardized count data
    # Z_cov_ex_hom shape: (#neurons x #neurons)

    Z_cov_ex_hom = nspu.cov_standardized_data(reshaped_Z_count_ex_hom, assume_centered=False)
    Z_cov_in_hom = nspu.cov_standardized_data(reshaped_Z_count_in_hom, assume_centered=False)

    # print size of covariance matrices: expect (neurons x neurons)
    print(f"Covariance ex matrix shape (#neurons x #neurons): {Z_cov_ex_hom.shape}")
    print(f"Covariance in matrix shape (#neurons x #neurons): {Z_cov_in_hom.shape}")

    # save the ex and in covariance matrices computed
    nspu.save_covariance_json(
        Z_cov_ex_hom,
        OUT / "ex_cov_homogeneous_poisson.json",
        label="excitatory_homogeneous_poisson",
    )

    nspu.save_covariance_json(
        Z_cov_in_hom,
        OUT / "in_cov_homogeneous_poisson.json",
        label="inhibitory_homogeneous_poisson",
    )

    # plot the ex and in covariance matrices computed
    nspu.plot_covariance_matrix(
        Z_cov_ex_hom,
        OUT / "batch_excitatory_standardized_homogeneous_poisson_covariance.png",
        title = "Excitatory Standardized Homogeneous Poisson Covariance Matrix",
    )

    nspu.plot_covariance_matrix(
        Z_cov_in_hom,
        OUT / "batch_inhibitory_standardized_homogeneous_poisson_covariance.png",
        title = "Inhibitory Standardized Homogeneous Poisson Covariance Matrix",
    )


    # create summary text files with the summary in addition to the json file summary and data
    print("Homogenous Assumption Covariance Matrix for Excitatory Network Summary:\n")
    nspu.summarize_covariance(
        Z_cov_ex_hom,
        OUT / "ex_homogenous_covariance_matrix_summary.txt",
        label="Excitatory"
    )

    print("Homogenous Assumption Covariance Matrix for Inhibitory Network Summary:\n")
    nspu.summarize_covariance(
        Z_cov_in_hom,
        OUT / f"in_homogenous_covariance_matrix_summary.txt",
        label="Inhibitory"
    )

    # TIME-SLICE Covariance Matrices - EXCITATORY - HOMOGENEOUS
    # create covariance matrices across time bins
    # only doing it for homogeneous Poisson assumption right now
    # for excitatory spike trains
    slice_dir = OUT / "time_slice_covariances_homogeneous_ex"

    # generate
    covs_ex_hom_slices, time_indices = nspu.covariance_by_time_slice(
        Z_count_ex_standardized_hom,
        step=10,
        start=0,
        assume_centered=False,
    )

    # save to files
    nspu.save_time_slice_covariances_json(
        covs_ex_hom_slices,
        time_indices,
        OUT / "ex_homogeneous_covariance_every_10th_time_slice.json",
        label="excitatory_homogeneous_every_10th_time_slice",
        bin_dt=bin_dt,
    )

    # plot each individual matrix across time slices
    nspu.plot_time_slice_covariances(
        covs_ex_hom_slices,
        time_indices,
        slice_dir,
        label="ex_homogeneous",
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet 20x4 shared scale
    # some matrices get washed out
    nspu.plot_covariance_contact_sheet(
        covs_ex_hom_slices,
        time_indices,
        OUT / "ex_20x4_shared_scale_homogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet using auto scale
    # where each image gets its own color scale so they display
    # like they would individually
    nspu.plot_covariance_contact_sheet_autoscale(
        covs_ex_hom_slices,
        time_indices,
        OUT / "ex_20x4_autoscale_homogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # Homogeneous Excitatory
    # Spectral Analysis: Eigenvalue / Eigenvector analysis
    #
    # 1. Leading Eigenvalues for all 50 neurons across time slices
    #    per neuron eigenvalue as a function of time
    #    (one image: #neurons x #timebins, e.g. 50x80)
    #
    # 2. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    USE ABSOLUTE VALUE for each coefficient, don't allow negative values
    #    which indicate direction change
    #
    # 3. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    DO NOT USE ABSOLUTE VALUE for each coefficient, allow for negative values

    nspu.spectral_analysis_covariances(
        covs_ex_hom_slices,
        time_indices,
        OUT / "spectral_homogeneous_ex",
        label="ex_homogeneous",
        bin_dt=bin_dt,
        n_leading=10,
    )

    # TIME-SLICE Covariance Matrices - INHIBITORY - HOMOGENEOUS
    # create covariance matrices across time bins
    # only doing it for homogeneous Poisson assumption right now
    # for inhibitory spike trains
    slice_dir = OUT / "time_slice_covariances_homogeneous_in"

    covs_in_hom_slices, time_indices = nspu.covariance_by_time_slice(
        Z_count_in_standardized_hom,
        step=10,
        start=0,
        assume_centered=False,
    )

    # save the covariances json file
    nspu.save_time_slice_covariances_json(
        covs_in_hom_slices,
        time_indices,
        OUT / "in_homogeneous_covariance_every_10th_time_slice.json",
        label="inhibitory_homogeneous_every_10th_time_slice",
        bin_dt=bin_dt,
    )

    # plots each individual covariance matrix for time slice
    nspu.plot_time_slice_covariances(
        covs_in_hom_slices,
        time_indices,
        slice_dir,
        label="in_homogeneous",
        bin_dt=bin_dt,
    )


    # plot the covariance contact sheet 20x4 shared scale
       # some matrices get washed out
    nspu.plot_covariance_contact_sheet(
        covs_in_hom_slices,
        time_indices,
        OUT / "in_20x4_shared_scale_homogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet using auto scale
    # where each image gets its own color scale so they display
    # like they would individually
    nspu.plot_covariance_contact_sheet_autoscale(
        covs_in_hom_slices,
        time_indices,
        OUT / "in_20x4_autoscale_homogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # Homogeneous Inhibitory
    # Spectral Analysis: Eigenvalue / Eigenvector analysis
    #
    # 1. Leading Eigenvalues for all 50 neurons across time slices
    #    per neuron eigenvalue as a function of time
    #    (one image: #neurons x #timebins, e.g. 50x80)
    #
    # 2. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    USE ABSOLUTE VALUE for each coefficient, don't allow negative values
    #    which indicate direction change
    #
    # 3. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    DO NOT USE ABSOLUTE VALUE for each coefficient, allow for negative values

    nspu.spectral_analysis_covariances(
        covs_in_hom_slices,
        time_indices,
        OUT / "spectral_homogeneous_in",
        label="in_homogeneous",
        bin_dt=bin_dt,
        n_leading=10,
    )

    #endregion

    #region Second Standardization: Inhomogeneous Poisson Assumption


    # Second Standardization: Inhomogeneous Poisson Assumption
    # INHOMOGENOUS POISSON ASSUMPTION (empirical mean, theoretical std)
    # Run the standardization using empirical mean and theoretical standard deviation
    Z_count_ex_standardized_inhom = nspu.standardize_counts(count_ex, mode="Inhomogeneous")
    Z_count_in_standardized_inhom = nspu.standardize_counts(count_in, mode="Inhomogeneous")

    # print shapes, expect: (runs, neurons, time-bins)
    print(f"Shape of Z score matrix of covariances under Homogeneous Poisson assumption - ex: {Z_count_ex_standardized_inhom.shape}")
    print(f"Shape of Z score matrix of covariances under Homogeneous Poisson assumption - in: {Z_count_in_standardized_inhom.shape}")

    # Reshape standardized counts for covariance matrix by
    #    combining spike data for neurons across all_observations
    reshaped_Z_count_ex_inhom = nspu.reshape_standardized_counts_for_covariance(Z_count_ex_standardized_inhom)
    reshaped_Z_count_in_inhom = nspu.reshape_standardized_counts_for_covariance(Z_count_in_standardized_inhom)

    print(reshaped_Z_count_ex_inhom.shape)
    print(reshaped_Z_count_in_inhom.shape)  # expect: (neurons, runs x bins = total_observations)

    # Generate the covariance matrix of Z scores for the standardized count data
    #  Z_cov_ex_hom shape: (#neurons x #neurons)
    Z_cov_ex_inhom = nspu.cov_standardized_data(reshaped_Z_count_ex_inhom, assume_centered=False)
    Z_cov_in_inhom = nspu.cov_standardized_data(reshaped_Z_count_in_inhom, assume_centered=False)

    # print size of covariance matrices: expect (neurons x neurons)
    print(f"Covariance ex matrix shape (#neurons x #neurons): {Z_cov_ex_inhom.shape}")
    print(f"Covariance in matrix shape (#neurons x #neurons): {Z_cov_in_inhom.shape}")

    # save the ex and in covariance matrices computed
    nspu.save_covariance_json(
        Z_cov_ex_inhom,
        OUT / "ex_cov_inhomogeneous_poisson.json",
        label="excitatory_inhomogeneous_poisson",
    )

    nspu.save_covariance_json(
        Z_cov_in_inhom,
        OUT / "in_cov_inhomogeneous_poisson.json",
        label="inhibitory_inhomogeneous_poisson",
    )

    # plot the ex and in covariance matrices computed
    nspu.plot_covariance_matrix(
        Z_cov_ex_inhom,
        OUT / "batch_excitatory_standardized_inhomogeneous_poisson_covariance.png",
        title = "Excitatory Standardized Inhomogeneous Poisson Covariance Matrix",
    )

    nspu.plot_covariance_matrix(
        Z_cov_in_inhom,
        OUT / "batch_inhibitory_standardized_inhomogeneous_poisson_covariance.png",
        title = "Inhibitory Standardized Inhomogeneous Poisson Covariance Matrix",
    )


    # create summary text files with the summary in addition to the json file summary and data
    print("Inhomogenous Assumption Covariance Matrix for Excitatory Network Summary:\n")
    nspu.summarize_covariance(
        Z_cov_ex_inhom,
        OUT / "ex_inhomogenous_covariance_matrix_summary.txt",
        label="Excitatory"
    )

    print("Inhomogenous Assumption Covariance Matrix for Inhibitory Network Summary:\n")
    nspu.summarize_covariance(
        Z_cov_in_inhom,
        OUT / "in_inhomogenous_covariance_matrix_summary.txt",
        label="Inhibitory"
    )

    # TIME-SLICE Covariance Matrices - EXCITATORY - INHOMOGENEOUS

    # create covariance matrices across time bins
    # only doing it for homogeneous Poisson assumption right now
    # for excitatory spike trains
    slice_dir = OUT / "time_slice_covariances_inhomogeneous_ex"

    # generate
    covs_ex_inhom_slices, time_indices = nspu.covariance_by_time_slice(
        Z_count_ex_standardized_inhom,
        step=10,
        start=0,
        assume_centered=False,
    )

    # save to files
    nspu.save_time_slice_covariances_json(
        covs_ex_inhom_slices,
        time_indices,
        OUT / "ex_inhomogeneous_covariance_every_10th_time_slice.json",
        label="excitatory_inhomogeneous_every_10th_time_slice",
        bin_dt=bin_dt,
    )

    # plot each individual matrix across time slices
    nspu.plot_time_slice_covariances(
        covs_ex_inhom_slices,
        time_indices,
        slice_dir,
        label="ex_inhomogeneous",
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet 20x4 shared scale
    # some matrices get washed out
    nspu.plot_covariance_contact_sheet(
        covs_ex_inhom_slices,
        time_indices,
        OUT / "ex_20x4_shared_scale_inhomogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet using auto scale
    # where each image gets its own color scale so they display
    # as they would individually
    nspu.plot_covariance_contact_sheet_autoscale(
        covs_ex_inhom_slices,
        time_indices,
        OUT / "ex_20x4_autoscale_inhomogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # Inhomogeneous Excitatory
    # Spectral Analysis: Eigenvalue / Eigenvector analysis
    #
    # 1. Leading Eigenvalues for all 50 neurons across time slices
    #    per neuron eigenvalue as a function of time
    #    (one image: #neurons x #timebins, e.g. 50x80)
    #
    # 2. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    USE ABSOLUTE VALUE for each coefficient, don't allow negative values
    #    which indicate direction change
    #
    # 3. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    DO NOT USE ABSOLUTE VALUE for each coefficient, allow for negative values
    nspu.spectral_analysis_covariances(
        covs_ex_inhom_slices,
        time_indices,
        OUT / "spectral_inhomogeneous_ex",
        label="ex_inhomogeneous",
        bin_dt=bin_dt,
        n_leading=10,
    )

    # TIME-SLICE Covariance Matrices - INHIBITORY - INHOMOGENEOUS
    # create covariance matrices across time bins
    # only doing it for homogeneous Poisson assumption right now
    # for inhibitory spike trains
    slice_dir = OUT / "time_slice_covariances_inhomogeneous_in"

    covs_in_inhom_slices, time_indices = nspu.covariance_by_time_slice(
        Z_count_in_standardized_inhom,
        step=10,
        start=0,
        assume_centered=False,
    )

    # save the covariances json file
    nspu.save_time_slice_covariances_json(
        covs_in_inhom_slices,
        time_indices,
        OUT / "in_inhomogeneous_covariance_every_10th_time_slice.json",
        label="inhibitory_inhomogeneous_every_10th_time_slice",
        bin_dt=bin_dt,
    )

    # plots each individual covariance matrix for time slice
    nspu.plot_time_slice_covariances(
        covs_in_inhom_slices,
        time_indices,
        slice_dir,
        label="in_inhomogeneous",
        bin_dt=bin_dt,
    )


    # plot the covariance contact sheet 20x4 shared scale
       # some matrices get washed out
    nspu.plot_covariance_contact_sheet(
        covs_in_inhom_slices,
        time_indices,
        OUT / "in_20x4_shared_scale_inhomogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet using auto scale
    # where each image gets its own color scale so they display
    # like they would individually
    nspu.plot_covariance_contact_sheet_autoscale(
        covs_in_inhom_slices,
        time_indices,
        OUT / "in_20x4_autoscale_inhomogeneous_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # Inhomogeneous Inhibitory
    # Spectral Analysis: Eigenvalue / Eigenvector analysis
    #
    # 1. Leading Eigenvalues for all 50 neurons across time slices
    #    per neuron eigenvalue as a function of time
    #    (one image: #neurons x #timebins, e.g. 50x80)
    #
    # 2. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    USE ABSOLUTE VALUE for each coefficient, don't allow negative values
    #    which indicate direction change
    #
    # 3. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    DO NOT USE ABSOLUTE VALUE for each coefficient, allow for negative values
    nspu.spectral_analysis_covariances(
        covs_in_inhom_slices,
        time_indices,
        OUT / "spectral_inhomogeneous_in",
        label="in_inhomogeneous",
        bin_dt=bin_dt,
        n_leading=10,
    )

    #endregion

    #region Third Standardization: Non-Poisson Assumption

    # Third Standardization: Non-Poisson Assumption
    # NON POISSON ASSUMPTION (empirical mean, empirical std)
    # Run the standardization using empirical mean and empirical standard deviation
    Z_count_ex_standardized_nonPoisson = nspu.standardize_counts(count_ex, mode="NonPoisson")
    Z_count_in_standardized_nonPoisson = nspu.standardize_counts(count_in, mode="NonPoisson")

    # print shapes, expect: (runs, neurons, time-bins)
    print(f"Shape of Z score matrix of covariances under Non Poisson assumption - ex: {Z_count_ex_standardized_nonPoisson.shape}")
    print(f"Shape of Z score matrix of covariances under Non Poisson assumption - in: {Z_count_in_standardized_nonPoisson.shape}")

    # Reshape standardized counts for covariance matrix by
    #    combining spike data for neurons across all_observations
    reshaped_Z_count_ex_nonPoisson = nspu.reshape_standardized_counts_for_covariance(Z_count_ex_standardized_nonPoisson)
    reshaped_Z_count_in_nonPoisson = nspu.reshape_standardized_counts_for_covariance(Z_count_in_standardized_nonPoisson)

    print(reshaped_Z_count_ex_nonPoisson.shape)
    print(reshaped_Z_count_in_nonPoisson.shape)  # expect: (neurons, runs x bins = total_observations)

    # Generate the covariance matrix of Z scores for the standardized count data
    #  Z_cov_ex_hom shape: (#neurons x #neurons)
    Z_cov_ex_nonPoisson = nspu.cov_standardized_data(reshaped_Z_count_ex_nonPoisson, assume_centered=False)
    Z_cov_in_nonPoisson = nspu.cov_standardized_data(reshaped_Z_count_in_nonPoisson, assume_centered=False)

    # print size of covariance matrices: expect (neurons x neurons)
    print(f"Covariance ex matrix shape (#neurons x #neurons): {Z_cov_ex_nonPoisson.shape}")
    print(f"Covariance in matrix shape (#neurons x #neurons): {Z_cov_in_nonPoisson.shape}")

    # save the ex and in covariance matrices computed
    nspu.save_covariance_json(
        Z_cov_ex_nonPoisson,
        OUT / "ex_cov_inhomogeneous_non_poisson.json",
        label="excitatory_inhomogeneous_non_poisson",
    )

    nspu.save_covariance_json(
        Z_cov_in_nonPoisson,
        OUT / "in_cov_inhomogeneous_non_poisson.json",
        label="inhibitory_inhomogeneous_non_poisson",
    )

    # plot the ex and in covariance matrices computed
    nspu.plot_covariance_matrix(
        Z_cov_ex_nonPoisson,
        OUT / "batch_excitatory_standardized_non_poisson_covariance.png",
        title = "Excitatory Standardized Non-Poisson Covariance Matrix",
    )

    nspu.plot_covariance_matrix(
        Z_cov_in_nonPoisson,
        OUT / "batch_inhibitory_standardized_non_poisson_covariance.png",
        title = "Inhibitory Standardized Non-Poisson Covariance Matrix",
    )

    # create summary text files with the summary in addition to the json file summary and data
    print("Non-Poisson Assumption Covariance Matrix for Excitatory Network Summary:\n")
    nspu.summarize_covariance(
        Z_cov_ex_nonPoisson,
        OUT / "ex_nonPoisson_covariance_matrix_summary.txt",
        label="Excitatory"
    )

    print("Non-Poisson Assumption Covariance Matrix for Inhibitory Network Summary:\n")
    nspu.summarize_covariance(
        Z_cov_in_nonPoisson,
        OUT / "in_nonPoisson_covariance_matrix_summary.txt",
        label="Inhibitory"
    )

    # TIME-SLICE Covariance Matrices - EXCITATORY - NON-POISSON
    # create covariance matrices across time bins
    # only doing it for homogeneous Poisson assumption right now
    # for excitatory spike trains
    slice_dir = OUT / "time_slice_covariances_nonPoisson_ex"

    # generate
    covs_ex_nonPoisson_slices, time_indices = nspu.covariance_by_time_slice(
        Z_count_ex_standardized_nonPoisson,
        step=10,
        start=0,
        assume_centered=False,
    )

    # save to files
    nspu.save_time_slice_covariances_json(
        covs_ex_nonPoisson_slices,
        time_indices,
        OUT / "ex_nonPoisson_covariance_every_10th_time_slice.json",
        label="excitatory_nonPoisson_every_10th_time_slice",
        bin_dt=bin_dt,
    )

    # plot each individual matrix across time slices
    nspu.plot_time_slice_covariances(
        covs_ex_nonPoisson_slices,
        time_indices,
        slice_dir,
        label="ex_nonPoisson",
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet 20x4 shared scale
    # some matrices get washed out
    nspu.plot_covariance_contact_sheet(
        covs_ex_nonPoisson_slices,
        time_indices,
        OUT / "ex_20x4_shared_scale_nonPoisson_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet using auto scale
    # where each image gets its own color scale so they display
    # like they would individually
    nspu.plot_covariance_contact_sheet_autoscale(
        covs_ex_nonPoisson_slices,
        time_indices,
        OUT / "ex_20x4_autoscale_nonPoisson_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # NonPoisson Excitatory
    # Spectral Analysis: Eigenvalue / Eigenvector analysis
    #
    # 1. Leading Eigenvalues for all 50 neurons across time slices
    #    per neuron eigenvalue as a function of time
    #    (one image: #neurons x #timebins, e.g. 50x80)
    #
    # 2. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    USE ABSOLUTE VALUE for each coefficient, don't allow negative values
    #    which indicate direction change
    #
    # 3. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    DO NOT USE ABSOLUTE VALUE for each coefficient, allow for negative values
    nspu.spectral_analysis_covariances(
        covs_ex_nonPoisson_slices,
        time_indices,
        OUT / "spectral_nonPoisson_ex",
        label="ex_NonPoisson",
        bin_dt=bin_dt,
        n_leading=10,
    )

    # TIME-SLICE Covariance Matrices - INHIBITORY - NON-POISSON
    # create covariance matrices across time bins
    # only doing it for homogeneous Poisson assumption right now
    # for inhibitory spike trains
    slice_dir = OUT / "time_slice_covariances_nonPoisson_in"

    covs_in_nonPoisson_slices, time_indices = nspu.covariance_by_time_slice(
        Z_count_in_standardized_nonPoisson,
        step=10,
        start=0,
        assume_centered=False,
    )

    # save the covariances json file
    nspu.save_time_slice_covariances_json(
        covs_in_nonPoisson_slices,
        time_indices,
        OUT / "in_nonPoission_covariance_every_10th_time_slice.json",
        label="inhibitory_nonPoisson_every_10th_time_slice",
        bin_dt=bin_dt,
    )

    # plots each individual covariance matrix for time slice
    nspu.plot_time_slice_covariances(
        covs_in_nonPoisson_slices,
        time_indices,
        slice_dir,
        label="in_nonPoisson",
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet 20x4 shared scale
    # some matrices get washed out
    nspu.plot_covariance_contact_sheet(
        covs_in_nonPoisson_slices,
        time_indices,
        OUT / "in_20x4_shared_scale_nonPoisson_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # plot the covariance contact sheet using auto scale
    # where each image gets its own color scale so they display
    # like they would individually
    nspu.plot_covariance_contact_sheet_autoscale(
        covs_in_nonPoisson_slices,
        time_indices,
        OUT / "in_20x4_autoscale_nonPoisson_covariance_contact_sheet.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
    )

    # NonPoisson Inhibitory
    # Spectral Analysis: Eigenvalue / Eigenvector analysis
    #
    # 1. Leading Eigenvalues for all 50 neurons across time slices
    #    per neuron eigenvalue as a function of time
    #    (one image: #neurons x #timebins, e.g. 50x80)
    #
    # 2. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    USE ABSOLUTE VALUE for each coefficient, don't allow negative values
    #    which indicate direction change
    #
    # 3. Leading Eigenvector for the leading Eigenvalue per time slice
    #    (#timebins images, 80 images if 80 timebins, where each image
    #     shows the leading eigenvector corresponding to the leading eigenvalue)
    #    DO NOT USE ABSOLUTE VALUE for each coefficient, allow for negative values
    nspu.spectral_analysis_covariances(
        covs_in_nonPoisson_slices,
        time_indices,
        OUT / "spectral_nonPoisson_in",
        label="in_NonPoisson",
        bin_dt=bin_dt,
        n_leading=10,
    )

    #endregion

    print("\nDone.")



# ======================================================================
if __name__ == "__main__":
    main()