# Utilities for plotting
import json
import numpy as np
import matplotlib.pyplot as plt


def cumulative_counts_tensor(all_runs_spike_trains, simtime, dt):
    """
    all_runs_spike_trains:
        [
            run0_spikes,  # [neurons][spike_times]
            run1_spikes,
            ...
        ]

    returns:
        counts with shape (num_runs, num_neurons, num_bins)
    """
    return np.array([
        cumulative_counts_matrix(run_spikes, simtime=simtime, dt=dt)
        for run_spikes in all_runs_spike_trains
    ])


def brian2_theoretical_mean(count):

    empirical_mean = count.mean(axis=0)

    final_count = empirical_mean[:, -1]

    num_bins = count.shape[2]

    theoretical_mean = np.zeros(
        (len(final_count), num_bins)
    )

    for i in range(len(final_count)):
        theoretical_mean[i] = (
            final_count[i] / num_bins
        ) * np.arange(num_bins)

    # theoretical_std_dev =sqrt(theoretical_mean)

    return theoretical_mean


# Plot covariance matrices by time slices across runs
def plot_time_slice_covariances(
    covariances,
    time_indices,
    output_dir,
    label,
    bin_dt=1.0,
    dpi=300,
):
    """
    Save one covariance matrix plot per selected time slice.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    for cov, t_idx in zip(covariances, time_indices):
        time_ms = t_idx * bin_dt

        output_path = output_dir / f"{label}_cov_timebin_{t_idx:04d}_time_{time_ms:.1f}ms.png"

        title = f"{label} Covariance Matrix at t={time_ms:.1f} ms"

        plot_covariance_matrix(
            cov,
            output_path,
            title=title,
        )


# Save covariance matrices by time slices across runs
def save_time_slice_covariances_json(
    covariances,
    time_indices,
    output_path,
    label=None,
    bin_dt=1.0,
):
    """
    Save all time-slice covariance matrices to one JSON file.
    """

    summary_data = {
        "label": label,
        "num_matrices": len(covariances),
        "time_indices": time_indices,
        "time_ms": [
            float(t * bin_dt)
            for t in time_indices
        ],
        "shape_each": list(covariances[0].shape) if covariances else None,
        "covariance_matrices": [
            cov.tolist()
            for cov in covariances
        ],
    }

    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=4)


# Covariance matrix by time slices across runs
def covariance_by_time_slice(Z, step=10, start=0, assume_centered=False):
    """
    Compute one neuron-neuron covariance matrix for each selected time slice.

    Z shape:
        runs x neurons x bins

    Returns:
        covariances: list of covariance matrices
        time_indices: list of selected time-bin indices
    """

    covariances = []
    time_indices = list(range(start, Z.shape[2], step))

    for t in time_indices:
        # Z_t shape: runs x neurons
        Z_t = Z[:, :, t]

        # X_t shape: neurons x runs
        X_t = Z_t.T

        cov_t = cov_standardized_data(
            X_t,
            assume_centered=assume_centered,
        )

        covariances.append(cov_t)

    return covariances, time_indices


# Summarize covariance matrix
# format: summary parameters followed by matrix data
def summarize_covariance(cov, output_path, label="Excitatory"):
    """
    Print and save covariance summary.
    """
    diag = np.diag(cov)
    off = cov[~np.eye(cov.shape[0], dtype=bool)]

    summary = (
        f"Covariance Matrix Summary ({label})\n"
        f"{'-'*40}\n"
        f"Diagonal mean: {diag.mean():.3f}\n"
        f"Diagonal std:  {diag.std():.3f}\n"
        f"Offdiag mean:  {off.mean():.3f}\n"
        f"Offdiag std:   {off.std():.3f}\n"
    )

    print(summary)

    with open(output_path, "w") as f:
        f.write(summary)


# Save covariance matrix to JSON
#   format: summary parameters followed by matrix data
def save_covariance_json(cov_matrix, output_path, label=None, params=None):
    """
    Save covariance matrix and summary to JSON.
    """

    diag = np.diag(cov_matrix)
    off = cov_matrix[~np.eye(cov_matrix.shape[0], dtype=bool)]

    matrix_summary = {
        "label": label,
        "shape": list(cov_matrix.shape),
        "summary": {
            "diagonal_mean": float(diag.mean()),
            "diagonal_std": float(diag.std()),
            "offdiag_mean": float(off.mean()),
            "offdiag_std": float(off.std()),
        },
        "params": params or {},
        # need to use .tolist() as NumPy arrays are not directly JSON serializable
        "covariance_matrix": cov_matrix.tolist(),
    }

    # print a summary to the terminal
    ## print(matrix_summary)

    # write to file
    with open(output_path, "w") as f:
        json.dump(matrix_summary, f, indent=4)


# Standardize counting process data:
#  ----- Z = (data_point - mean) / standard deviation
#
# note: need to drop 1st column - t=0
#  ----- otherwise we get an error: divide by zero encountered in divide
#  ----- that's t=0 in the time bins, everything is 0
def standardize_counts(counts, mode):
    # Mode determines how we standardize the data:
    # - Theoretical assumes Homogeneous Poisson and uses theoretical mean and standard deviation
    # - Empirical assumes Inhomogenous Poisson and uses empirical mean and standard deviation
    # - NotPoisson assumes the data is not Poisson Distributed
    #
    # counts: num_runs x num_neurons x num_bins
    num_runs, num_neurons, num_bins = counts.shape

    # generates the mean for each neuron across all runs per time bin
    empirical_mean = counts.mean(axis=0)            # neurons x bins (runs are all averaged out)
    empirical_std = counts.std(axis=0, ddof=1)      # neurons x bins:
                                                    # ddof = 0 population standard deviation
                                                    # ddof = 1 sample standard deviation
                                                    # use "1" since we have 50 samples from entire population
    sqrt_empirical_mean = np.sqrt(empirical_mean)   # this will be used in Inhomogeneous standardization

    final_count = empirical_mean[:, -1]   # neurons (final cumulative spike count per neuron)
                                          # [24, 31, 18 etc.]

    # create a time axis with bin indices [0, 1, 2, ..., num_bins-1]
    bin_axis = np.arange(num_bins)

    # for a homogenous Poisson process, expected count grows linearly over time
    # calculate the theoretical mean standard deviation
    # for each neuron per time bin
    # ---- expected count(t) for each neuron
    #
    # final_count
    # [24,31,18, ...] turns into final_count [[24],[31],[18], ...]
    # shape (50,1) # N_rec x 1
    # bin_axis
    # [0,1,2,...] turns into [[0,1,2,...]]
    # shape (1, 8000)
    # theoretical mean shape=  (50,1) x (1,8000) = (50,8000)
    # theoretical_mean for each neuron per  time_bin based on Poisson assumption
    # assuming linear growth, starting at 0 count at t0,
    # increment final_count/num_bins at each time bin (hence: * bin_axis)
    theoretical_mean = (
        final_count[:, None] / num_bins
    ) * bin_axis[None, :]

    # theoretical standard deviation
    # (N_rec, time_bins) per neuron per time bin
    theoretical_std = np.sqrt(theoretical_mean)

    # drop index 0 from all since
    # t=0 before division - to avoid division by zero
    #      at t=0: expected count = 0, std = 0

    counts_nozero = counts[:, :, 1:]      # new shape (runs, N_rec, time_bins -1)

    theoretical_mean_nozero = theoretical_mean[:, 1:] # new shape (N_rec, time_bins -1)
    theoretical_std_nozero = theoretical_std[:, 1:]   # new shape (N_rec, time_bins -1)

    empirical_mean_nozero = empirical_mean[:, 1:]
    empirical_std_nozero = empirical_std[:,1:]
    sqrt_empirical_mean_nozero = sqrt_empirical_mean[:,1:]

    # Z score = (count - mean) / standard deviation
    #   for example, for 'Theoretical',
    #   the new count shows how many standard deviations above (+) or below (-)
    #   what a Homogeneous Poisson process predicts the count should be
    # -- inhibitory spike train on the NonPoisson assumption produced some
    # -- quiet neurons where we got division by zero since our std = 0
    # -- used safe_std instead for all
    if mode == "Homogeneous":  # (value-theoretical mean) / theoretical_std
        # use theoretical mean and standard deviation to standardize
        safe_std = theoretical_std_nozero.copy() # make a copy of the standard deviation array
        safe_std[safe_std == 0] = np.nan #replace all 0 entries with nan to avoid division by zero
        Z = (counts_nozero - theoretical_mean_nozero) / safe_std
        Z = np.nan_to_num(Z, nan=0.0)   # replace any "nan"s with zero
                                        # std = 0 because every run produced 0 deviation from the mean
                                        # so, it's safe to replace those values with 0 instead of
                                        # leaving them as nan and compounding the nan
                                        # to the corresponding row/column in the matrix (displays as white row/colums)

    elif mode == "Inhomogeneous": # (value-empirical mean) / sqrt (empirical_mean)
        # use empirical mean and theoretical standard deviation to standardize
        safe_std = sqrt_empirical_mean_nozero.copy()    # make a copy of the array
        safe_std[safe_std == 0] = np.nan                # replace all 0 entries with nan to avoid division by 0
        Z = (counts_nozero - empirical_mean_nozero) / safe_std
        Z = np.nan_to_num(Z, nan=0.0)                   # replace any "nan"s with zero

    elif mode == "NonPoisson": # (value-empirical mean) / empirical_std
        # assume the distribution is not Poisson - use all empirical data
        # >>> note: on the inhibitory train particularly some neurons are quiet
        #       - hence std is zero
        #       - to avoid division by zero use safe_std as defined below
        safe_std = empirical_std_nozero.copy()  # make a copy of the standard deviation array
        safe_std[safe_std == 0] = np.nan        # replace all 0 entries with nan to avoid division by zero
        Z = (counts_nozero - empirical_mean_nozero) / safe_std
        Z = np.nan_to_num(Z, nan=0.0)           # replace any "nan"s with zero

    else:
        print(f"Non expected standardization parameter: {mode}\n")


    # Z shape is (num_runs, N_rec, time_bins)
    # Z[run, neuron, time] = standardized data as described above

    return Z


# reshape standard counts to run across all data points for all runs and time bins
def reshape_standardized_counts_for_covariance(Z):
    """
    Z shape:
        (num_runs, num_neurons, num_bins)

    returns:
        X shape:
        (num_neurons, num_runs * num_bins)
    """

    # Z[run][neuron][time] = contains a standardized Poisson deviation
    # transpose: np.transpose(Z, (1,0,2))
    #      rearranges: old axis 1 is now 0, old axis 0 is now 1, 2 unchanged
    # old: (run, neuron, time)
    # new: (neuron, run, time)

    # num_observations = num_runs x num_bins
    #     reshape (num_neurons, num_observations)
    # Z.shape[1] = number of neurons
    # reshape (Z.shape[1], -1): keep num_neurons, flatten everything else (runs x time_bins)
    #
    return np.transpose(Z, (1, 0, 2)).reshape(Z.shape[1], -1)


def covariance_every_xth_timebin(Z, step=10, start=0, assume_centered=False):
    """
    Compute neuron-neuron covariance using every step-th time bin.

    Z shape:
        (num_runs, num_neurons, num_bins)

    Returns:
        cov shape:
        (num_neurons, num_neurons)
    """

    Z_sliced = Z[:, :, start::step]

    X = np.transpose(Z_sliced, (1, 0, 2)).reshape(Z_sliced.shape[1], -1)

    cov = cov_standardized_data(X, assume_centered=assume_centered)

    return cov


def covariance_from_standardized_runs(Z):
    """
    Z shape:
        (num_runs, num_neurons, num_bins)

    returns:
        neuron x neuron covariance matrix
    """

    num_runs, num_neurons, num_bins = Z.shape

    # combine runs and time into one observation axis
    X = Z.transpose(1, 0, 2).reshape(num_neurons, num_runs * num_bins)

    # center each neuron row
    X = X - X.mean(axis=1, keepdims=True)

    return X @ X.T / (X.shape[1] - 1)


def plot_covariance_matrix (cov_matrix, output_path, title="Standardized Homogeneous Poisson Covariance Matrix"):
    plt.figure(figsize=(7, 6))
    plt.imshow(cov_matrix, aspect="equal")
    plt.colorbar(label="Covariance")
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


#  Brian2 equivalent
#  with added error checking and option to center
def cov_standardized_data(X, assume_centered=True):
    """
    Compute covariance matrix for standardized data.

    Parameters
    ----------
    X : array-like
        Matrix with shape (neurons, timepoints).

    assume_centered : bool
        If True, assumes each row already has mean 0.
        If False, subtracts row means first.

    Returns
    -------
    cov : ndarray
        Neuron-by-neuron covariance matrix.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape neurons × timepoints.")

    if X.shape[1] < 2:
        raise ValueError("X must have at least two timepoints.")

    if not assume_centered:
        X = X - X.mean(axis=1, keepdims=True)

    return X @ X.T / (X.shape[1] - 1)


def load_spike_trains(path):
    with open(path, "r") as f:
        return json.load(f)


def flatten_spike_trains(spike_trains):
    if not spike_trains:
        return np.array([])
    return np.concatenate([np.array(train) for train in spike_trains if len(train) > 0])


# load the run parameters mainly to access simtime and t0 time shift
# when analyzing the results
# since all of the run parameters are written to a json file
# it is better to read them from that file instead of
# loading the initial config file - in case some changes
# were made during the runtime of the simulation
def load_sim_params(prefix, output_dir):
    import json
    from pathlib import Path

    param_file = Path(output_dir) / f"{prefix}_params.json"

    with open(param_file, "r") as f:
        params = json.load(f)

    return params


def spike_scatter_plot (
    excitatory_spike_trains, # after the warm-up period is discarded
    inhibitory_spike_trains, # for both spike trains
    output_path,
    t_start, #=0.0,
    t_end, # =1000.0,
    n_plot, # =25,
    bin_width, #=1.0,
    rate_ylim, # =(0, 200),
):
    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1])
    ax_spikes, ax_rates = gs.subplots(sharex=True)

    # Raster: inhibitory neurons  -- these will be displated at the bottom
    for i, train in enumerate(inhibitory_spike_trains[:n_plot]):
        times = [t for t in train if t_start <= t <= t_end]
        ax_spikes.plot(times, [i] * len(times), "|", markersize=3)

    gap = 5 # leave a gap in between neuron types

    # Raster: excitatory neurons
    offset = n_plot + 5
    for i, train in enumerate(excitatory_spike_trains[:n_plot]):
        times = [t for t in train if t_start <= t <= t_end]
        ax_spikes.plot(times, [i + offset] * len(times), "|", markersize=3)

    # ax_spikes.set_ylabel("Neuron")
    ax_spikes.set_ylabel("Inhibitory   /    Excitatory (top) Neurons")
    ax_spikes.set_yticks([])
    ax_spikes.set_title("Spike raster")

    # Population rate
    bins = np.arange(t_start, t_end + bin_width, bin_width)

    ex_all = flatten_spike_trains(excitatory_spike_trains)
    in_all = flatten_spike_trains(inhibitory_spike_trains)

    ex_all = ex_all[(ex_all >= t_start) & (ex_all <= t_end)]
    in_all = in_all[(in_all >= t_start) & (in_all <= t_end)]

    ex_counts, _ = np.histogram(ex_all, bins=bins)
    in_counts, _ = np.histogram(in_all, bins=bins)

    ex_rate = ex_counts / len(excitatory_spike_trains) / (bin_width / t_end) # bin_width / 1000.0
    in_rate = in_counts / len(inhibitory_spike_trains) / (bin_width / t_end)

    bin_centers = bins[:-1]

    ax_rates.plot(bin_centers, ex_rate, label="Excitatory")
    ax_rates.plot(bin_centers, in_rate, label="Inhibitory")

    ax_rates.set_ylabel("Rate [Hz]")
    ax_rates.set_xlabel("Time [ms]")

    if rate_ylim is not None:
        ax_rates.set_ylim(rate_ylim)
    else:
        max_rate = max(ex_rate.max(), in_rate.max())

    ax_rates.set_ylim(0, max_rate * 1.1)
    ax_rates.set_ylim(rate_ylim)
    ax_rates.legend(fontsize=6, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_network_mean_std(spike_trains, simtime, dt=1.0):
    matrix = spike_train_to_binary_matrix(spike_trains, simtime, dt)

    mean_over_neurons = matrix.mean(axis=0)
    std_over_neurons = matrix.std(axis=0)

    time_axis = np.arange(matrix.shape[1]) * dt

    return time_axis, mean_over_neurons, std_over_neurons


def standardize_signal(x):
    x = np.asarray(x)

    if x.std() == 0:
        return np.zeros_like(x)

    return (x - x.mean()) / x.std()


def plot_mean_std_vs_time(
    spike_trains,
    output_path,
    simtime,
    dt=1.0,
    title="Excitatory Network Mean and Standard Deviation vs Time",
):
    time_axis, mean_signal, std_signal = compute_network_mean_std(
        spike_trains,
        simtime,
        dt,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, mean_signal, label="Mean")
    plt.plot(time_axis, std_signal, label="Standard deviation")
    plt.xlabel("Time [ms]")
    plt.ylabel("Spike probability per bin")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_standardized_mean_std_vs_time(
    spike_trains,
    output_path,
    simtime,
    dt=1.0,
    title="Standardized Excitatory Network Mean and Standard Deviation vs Time",
):
    time_axis, mean_signal, std_signal = compute_network_mean_std(
        spike_trains,
        simtime,
        dt,
    )

    mean_z = standardize_signal(mean_signal)
    std_z = standardize_signal(std_signal)

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, mean_z, label="Standardized mean")
    plt.plot(time_axis, std_z, label="Standardized standard deviation")
    plt.xlabel("Time [ms]")
    plt.ylabel("z-score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# cumulative counts
def cumulative_counts_matrix(spike_trains, simtime, dt=1.0):
    num_neurons = len(spike_trains)
    num_bins = int(simtime / dt)

    counts = np.zeros((num_neurons, num_bins))

    for neuron_idx, train in enumerate(spike_trains):
        for spike_time in train:
            bin_idx = int(spike_time / dt)
            if 0 <= bin_idx < num_bins:
                counts[neuron_idx, bin_idx] += 1

    return np.cumsum(counts, axis=1)


def standardized_homogeneous_poisson_matrix(spike_trains, simtime, dt):
    """
    Convert spike trains to standardized cumulative counts under

    a homogeneous Poisson assumption. This matches the Brian2 11.19 sim
    input line [89]

    Output shape:
        neurons x time_bins
    """

    counts = cumulative_counts_matrix(spike_trains, simtime=simtime, dt=dt)

    empirical_mean = counts.mean(axis=0)

    time_axis = np.arange(counts.shape[1]) * dt
    time_sec = time_axis / 1000.0

    # Estimate lambda from final cumulative count
    rate_hz = empirical_mean[-1] / time_sec[-1]

    theoretical_mean = rate_hz * time_sec
    theoretical_std = np.sqrt(theoretical_mean)

    # Avoid division by zero at t=0
    theoretical_std[theoretical_std == 0] = np.nan

    standardized = (counts - theoretical_mean) / theoretical_std

    # Drop t=0 column
    standardized = standardized[:, 1:]

    return standardized


def homogeneous_poisson_covariance_matrix(standardized_matrix):
    """
    Compute covariance matrix of standardized cumulative spike counts.

    Output shape:
        neurons x neurons
    """
    cov_matrix = np.cov(standardized_matrix)

    return cov_matrix


def plot2_homogeneous_poisson_covariance_matrix(spike_trains, output_path, simtime, dt):
    cov_matrix = homogeneous_poisson_covariance_matrix(
        spike_trains,
        simtime=simtime,
        dt=dt,
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(cov_matrix, aspect="auto")
    plt.colorbar(label="Covariance")
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.title("Standardized Homogeneous Poisson Covariance Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cov_matrix

def plot_homogeneous_single_run_poisson_covariance_matrix(cov_matrix, output_path):

    plt.figure(figsize=(7, 6))
    plt.imshow(cov_matrix, aspect="auto")
    plt.colorbar(label="Covariance")
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.title("Standardized Homogeneous Poisson Covariance Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_allruns_spike_trains(prefixes, output_dir, population="excitatory"):
    """
    Load spike trains from many simulation runs.

    Returns:
        all_runs_spikes[run][neuron] = [spike_time_1, spike_time_2, ...]
    """
    all_runs_spikes = []

    for prefix in prefixes:
        spike_file = output_dir / f"{prefix}_{population}_spikes.json"

        if not spike_file.exists():
            print(f"Missing file: {spike_file}")
            continue

        spike_trains = load_spike_trains(spike_file)
        all_runs_spikes.append(spike_trains)

    return all_runs_spikes


# the initial warmup time (time=t0) needs to be disregarded
def discard_warmup_spikes(spike_trains, t0, shift_time=False):
    """
    Remove spikes before t0 during analysis.

    spike_trains:
        [
            [t, t, t, ...],   # neuron 0
            [t, t, t, ...],   # neuron 1
            ...
        ]

    shift_time:
        False -> keep original times, e.g. 250 ms stays 250 ms
        True  -> subtract t0, e.g. 250 ms becomes 50 ms
    """

    filtered = []

    for neuron_spikes in spike_trains:
        if shift_time:
            # append to filtered all spike times that are >= t0
            # and shift the new time stamps to subtract t0
            filtered.append([t - t0 for t in neuron_spikes if t >= t0])
        else:
            # append all spike times >= t0 with their current run times
            filtered.append([t for t in neuron_spikes if t >= t0])

    return filtered


# discard the initial warm up period for all runs
def discard_warmup_all_runs(all_runs_spikes, t0, shift_time=True):
    """
    Apply warmup removal to every run.

    Input:
        all_runs_spikes[run][neuron] = [spike times]

    Output:
        same structure, but with spikes before t0 removed.
    """
    return [
        discard_warmup_spikes(run_spikes, t0=t0, shift_time=shift_time)
        for run_spikes in all_runs_spikes
    ]



# convert spike train to binary matrix with
#   1s in timeslots where is a spike
#   0s elsewhere
def spike_train_to_binary_matrix(spike_trains, simtime, dt=1.0):
    """
    Convert spike trains to matrix:
        rows = neurons
        columns = time bins
    """
    num_neurons = len(spike_trains)
    num_bins = int(simtime / dt)

    # load a matrix of size num_neurons x num_bins with 0s
    matrix = np.zeros((num_neurons, num_bins), dtype=float)

    # replace the entry with 1 if it falls within a delta-t
    for neuron_idx, train in enumerate(spike_trains):
        for spike_time in train:
            bin_idx = int(spike_time / dt)
            if 0 <= bin_idx < num_bins:
                matrix[neuron_idx, bin_idx] = 1.0

    return matrix


# convert all runs' spike trains to multi dimensional binary arrays
def spike_runs_to_binary_tensor(all_runs_spikes, simtime, dt):
    """
    Convert many runs into binary spike tensor.

    Input:
        all_runs_spikes[run][neuron] = [spike times]

    Output:
        binary_tensor shape = (num_runs, num_neurons, num_bins)
    """
    # ----- old version doesn't check for
    # ----- mis-match run dimension data in output directory
    # ----- added the error check

    binary_runs = []
    #for run_spikes in all_runs_spikes:
    #    binary_matrix = spike_train_to_binary_matrix(
    #        run_spikes,
    #        simtime=simtime,
    #        dt=dt,
    #    )
    #    binary_runs.append(binary_matrix)
    # return np.array(binary_runs)

    expected_shape = None

    # (num_runs, num_neuronsm num_bins) need to match
    # across runs for this work....
    # add another dimension to check the correct shape
    for run_idx, run_spikes in enumerate(all_runs_spikes):
        binary_matrix = spike_train_to_binary_matrix(
            run_spikes,
            simtime=simtime,
            dt=dt,
        )

        if expected_shape is None:
            expected_shape = binary_matrix.shape
        elif binary_matrix.shape != expected_shape:
            raise ValueError(
                f"Run {run_idx} has shape {binary_matrix.shape}, "
                f"expected {expected_shape}. "
                "You probably mixed runs with different N_rec/simtime/dt."
            )

        binary_runs.append(binary_matrix)

    return np.stack(binary_runs, axis=0)


# Counting Process
# apply to binary tensors: cumulative spike count up to the bin
def binary_tensor_to_counting_process(binary_tensor):
    """
    Convert binary spikes to cumulative counting process.

    Input:
        binary_tensor[run, neuron, time_bin] = 0 or 1

    Output:
        count_tensor[run, neuron, time_bin] = cumulative spike count up to that bin
    """
    return np.cumsum(binary_tensor, axis=2)

# generate a contact sheet for covariance matrices
def plot_covariance_contact_sheet(
    covariances,
    time_indices,
    output_path,
    nrows=20,
    ncols=4,
    bin_dt=1.0,
    vmin=None,
    vmax=None,
):
    """
    Plot many covariance matrices in one image.

    covariances:
        list of 2D covariance matrices

    time_indices:
        list of time-bin indices corresponding to covariances

    output_path:
        path to save the combined image
    """

    import math

    num_plots = len(covariances)

    if nrows * ncols < num_plots:
        raise ValueError(
            f"Grid {nrows}x{ncols} has only {nrows*ncols} slots "
            f"but needs {num_plots}."
        )

    # Use shared color scale unless explicitly supplied
    if vmin is None:
        vmin = min(np.nanmin(cov) for cov in covariances)
    if vmax is None:
        vmax = max(np.nanmax(cov) for cov in covariances)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.0, nrows * 2.0),
    )

    axes = np.asarray(axes).reshape(-1)

    last_im = None

    for ax_idx, ax in enumerate(axes):
        ax.axis("off")

        if ax_idx >= num_plots:
            continue

        cov = covariances[ax_idx]
        t_idx = time_indices[ax_idx]
        time_ms = t_idx * bin_dt

        last_im = ax.imshow(
            cov,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )

        # Tiny title only; remove this line if you want zero labels
        ax.set_title(f"{time_ms:.0f} ms", fontsize=6)

    # One shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.tolist(),
            shrink=0.6,
            fraction=0.02,
            pad=0.01,
        )
        cbar.ax.tick_params(labelsize=6)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# this contact sheet uses one colorscale per graph so images don't get washed out
def plot_covariance_contact_sheet_autoscale(
    covariances,
    time_indices,
    output_path,
    nrows=20,
    ncols=4,
    bin_dt=1.0,
):
    num_plots = len(covariances)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.0, nrows * 2.0),
    )

    axes = np.asarray(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        ax.axis("off")

        if ax_idx >= num_plots:
            continue

        cov = covariances[ax_idx]
        t_idx = time_indices[ax_idx]
        time_ms = t_idx * bin_dt

        # autoscale each matrix independently
        ax.imshow(cov, aspect="equal")

        ax.set_title(f"{time_ms:.0f} ms", fontsize=6)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Eigenvalue / Eigenvector Spectral Analysis
def spectral_analysis_eigen ():
    pass


# plot eigenvector contact sheet
def plot_eigenvector_contact_sheet(
    leading_vectors,
    time_indices,
    output_path,
    nrows=20,
    ncols=4,
    bin_dt=1.0,
    use_abs=False,
    shared_scale=True,
):
    """
    Plot leading eigenvectors across time slices as one contact sheet.

    leading_vecs shape:
        time_slices x neurons
    """

    data = np.abs(leading_vectors) if use_abs else leading_vectors

    num_plots = data.shape[0]

    if nrows * ncols < num_plots:
        raise ValueError(
            f"Grid {nrows}x{ncols} has only {nrows*ncols} slots "
            f"but needs {num_plots}."
        )

    if shared_scale:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    else:
        vmin = None
        vmax = None

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.0, nrows * 1.8),
    )

    axes = np.asarray(axes).reshape(-1)

    last_im = None

    for ax_idx, ax in enumerate(axes):
        ax.axis("off")

        if ax_idx >= num_plots:
            continue

        t_idx = time_indices[ax_idx]
        time_ms = t_idx * bin_dt

        vec = data[ax_idx, :].reshape(-1, 1)

        if shared_scale:
            last_im = ax.imshow(
                vec,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            last_im = ax.imshow(
                vec,
                aspect="auto",
            )

        ax.set_title(f"{time_ms:.0f} ms", fontsize=6)

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.tolist(),
            shrink=0.6,
            fraction=0.02,
            pad=0.01,
        )
        cbar.ax.tick_params(labelsize=6)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# PLOT leading eigenvalues for the covariance matrices
def plot_leading_eigenvalues(
    eigenvalues,
    time_indices,
    output_path,
    n_leading=10,
    bin_dt=1.0,
):
    """
    eigvals shape:
        time_slices x neurons
    """

    time_ms = np.array(time_indices) * bin_dt

    plt.figure(figsize=(9, 5))

    for k in range(min(n_leading, eigenvalues.shape[1])):
        plt.plot(time_ms, eigenvalues[:, k], label=f"lambda {k+1}")

    plt.xlabel("Time (ms)")
    plt.ylabel("Eigenvalue")
    plt.title("Leading covariance eigenvalues over time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



# PLOT leading eigenvalectors for the covariance matrices across time slices
def plot_leading_eigenvectors_by_time(
    leading_vectors,
    time_indices,
    output_dir,
    label,
    bin_dt=1.0,
    use_abs=False,
):
    """
    Save one image per time slice.

    leading_vecs shape:
        time_slices x neurons
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.abs(leading_vectors) if use_abs else leading_vectors

    # shared color scale for comparability
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    for i, t_idx in enumerate(time_indices):
        time_ms = t_idx * bin_dt

        vec = data[i, :].reshape(-1, 1)   # neurons x 1 image

        plt.figure(figsize=(2.5, 7))
        plt.imshow(vec, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Eigenvector coefficient")
        plt.xlabel("Leading eigenvector")
        plt.ylabel("Neuron")
        plt.title(f"{label}: t={time_ms:.1f} ms")
        plt.tight_layout()

        output_path = output_dir / f"{label}_leading_eigenvector_timebin_{t_idx:04d}_time_{time_ms:.1f}ms.png"

        plt.savefig(output_path, dpi=300)
        plt.close()


# plot leading eigenvector colormap
# rows    = neurons
# columns = time slices
# entry   = coefficient of neuron i in the leading eigenvector at time t
def plot_leading_eigenvector_colormap(
    leading_vecs,
    time_indices,
    output_path,
    label,
    bin_dt=1.0,
    use_abs=True,
):
    """
    Plot leading eigenvector coefficients as a neuron x time colormap.

    leading_vecs shape:
        time_slices x neurons

    Output image:
        neurons x time_slices
    """

    data = np.abs(leading_vecs) if use_abs else leading_vecs

    # transpose so rows = neurons, columns = time slices
    heatmap = data.T

    time_ms = np.array(time_indices) * bin_dt

    plt.figure(figsize=(10, 7))

    plt.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
    )

    plt.colorbar(label="Leading eigenvector coefficient")

    plt.xlabel("Time slice")
    plt.ylabel("Neuron")
    plt.title(label)

    # optional readable x-ticks
    tick_positions = np.linspace(0, len(time_indices) - 1, 8, dtype=int)
    plt.xticks(
        tick_positions,
        [f"{time_ms[i]:.0f}" for i in tick_positions],
        rotation=45,
    )
    plt.xlabel("Time (ms)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# Plot the total eigenvalue variance as a function of time
# both (signed) and (absolute value)
def plot_total_eigenvalue_variance(
    eigenvalues,
    time_indices,
    output_dir,
    label,
    bin_dt=1.0,
    modes=("signed", "absolute"),
):
    """
    Plot total covariance variance across time slices.

    eigenvalues shape:
        time_slices x neurons

    modes:
        "signed"   -> sum(lambda_i)
        "absolute" -> sum(abs(lambda_i))
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    time_ms = np.array(time_indices) * bin_dt

    for mode in modes:

        if mode == "signed":
            total_variance = np.sum(eigenvalues, axis=1)
            ylabel = "Signed sum of eigenvalues"
            title = f"{label}: Signed total eigenvalue variance"
            filename = f"{label}_total_eigenvalue_variance_signed.png"

        elif mode == "absolute":
            total_variance = np.sum(np.abs(eigenvalues), axis=1)
            ylabel = "Sum of absolute eigenvalues"
            title = f"{label}: Absolute total eigenvalue variance"
            filename = f"{label}_total_eigenvalue_variance_absolute.png"

        else:
            raise ValueError(f"Unknown mode: {mode}")

        plt.figure(figsize=(9, 5))
        plt.plot(time_ms, total_variance)
        plt.xlabel("Time (ms)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()



# plot the eigenvalues per time index in a color map
def plot_eigenvalue_spectrum_colormap(
    eigenvalues,
    time_indices,
    output_path,
    label,
    bin_dt=1.0,
    use_log=False,
    normalize_by_trace=False,
):
    """
    Plot eigenvalue spectrum across time.

    eigenvalues shape:
        time_slices x neurons

    Output:
        rows    = eigenvalue rank
        columns = time slices
        color   = eigenvalue magnitude
    """

    eig = np.array(eigenvalues, dtype=float)

    if normalize_by_trace:
        trace = np.sum(eig, axis=1, keepdims=True)
        trace[trace == 0] = np.nan
        eig = eig / trace

    if use_log:
        eig = np.log10(np.maximum(eig, 1e-12))

    heatmap = eig.T  # eigenvalue rank x time

    time_ms = np.array(time_indices) * bin_dt

    plt.figure(figsize=(10, 7))
    plt.imshow(
        heatmap,
        aspect="auto",
        origin="upper",
    )

    cbar_label = "Eigenvalue"
    if normalize_by_trace:
        cbar_label = "Fraction of total variance"
    if use_log:
        cbar_label = "log10(" + cbar_label + ")"

    plt.colorbar(label=cbar_label)

    plt.ylabel("Eigenvalue rank")
    plt.xlabel("Time (ms)")
    plt.title(label)

    tick_positions = np.linspace(0, len(time_indices) - 1, 8, dtype=int)
    plt.xticks(
        tick_positions,
        [f"{time_ms[i]:.0f}" for i in tick_positions],
        rotation=45,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# SPECTRAL ANALYSIS Covariances
def spectral_analysis_covariances(
    covariances,
    time_indices,
    output_dir,
    label,
    bin_dt=1.0,
    n_leading=10,
):
    """
    Compute eigenvalues/eigenvectors for covariance matrices across time slices.

    Saves:
        1 JSON file with all eigenvalues and eigenvectors
        1 leading-eigenvalue plot showing top eigenvalues across time
        80 leading-eigenvector images
        80 absolute-leading-eigenvector images
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    eigvals_all = []
    eigvecs_all = []
    leading_vecs = []

    for cov in covariances:
        # eigh is for symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(cov)

        # sort descending and reorder the eigenvectors to match
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals_all.append(eigvals)
        eigvecs_all.append(eigvecs)

        # leading eigenvector corresponds to largest eigenvalue (which is at index 0)
        # store the corresponding vector in leading_vecs array
        leading_vecs.append(eigvecs[:, 0])

    # store the eigenvalues for this one time slice to the list of all
    # after looping through all time slices, convert the lists to arrays
    eigvals_all = np.array(eigvals_all)       # time_slices x neurons
    eigvecs_all = np.array(eigvecs_all)       # time_slices x neurons x neurons
    leading_vecs = np.array(leading_vecs)     # time_slices x neurons

    # Save everything in one JSON
    summary_data = {
        "label": label,
        "num_time_slices": len(time_indices),
        "time_indices": time_indices,
        "time_ms": [float(t * bin_dt) for t in time_indices],
        "eigenvalues_shape": list(eigvals_all.shape),
        "eigenvectors_shape": list(eigvecs_all.shape),
        "leading_eigenvectors_shape": list(leading_vecs.shape),
        "eigenvalues": eigvals_all.tolist(),
        "eigenvectors": eigvecs_all.tolist(),
        "leading_eigenvectors": leading_vecs.tolist(),
        "leading_eigenvectors_abs": np.abs(leading_vecs).tolist(),
    }

    with open(output_dir / f"{label}_spectral_data.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    # Plot leading eigenvalues over time
    # for example: top 10 leading eigenvalues
    plot_leading_eigenvalues(
        eigvals_all,
        time_indices,
        output_dir / f"{label}_leading_eigenvalues_over_time.png",
        n_leading=n_leading,
        bin_dt=bin_dt,
    )

    # Plot leading eigenvalue variance
    plot_total_eigenvalue_variance(
        eigvals_all,
        time_indices,
        output_dir,
        label=label,
        bin_dt=bin_dt,
        modes=("signed", "absolute"),
    )

    # Plot all eigenvalues in a heatmap across time slices
    plot_eigenvalue_spectrum_colormap(
        eigvals_all,
        time_indices,
        output_dir / f"{label}_eigenvalue_spectrum_colormap.png",
        label=f"{label}: Eigenvalue spectrum over time",
        bin_dt=bin_dt,
        use_log=False,
        normalize_by_trace=False,
    )


    # Plot all normalized-eigenvalues in a heatmap across time slices
    plot_eigenvalue_spectrum_colormap(
        eigvals_all,
        time_indices,
        output_dir / f"{label}_eigenvalue_spectrum_fraction_variance_colormap.png",
        label=f"{label}: Fractional eigenvalue spectrum over time",
        bin_dt=bin_dt,
        use_log=False,
        normalize_by_trace=True,
    )


    # Plot leading eigenvector images
    plot_leading_eigenvectors_by_time(
        leading_vecs,
        time_indices,
        output_dir / "leading_eigenvectors_signed",
        label=f"{label}_signed",
        bin_dt=bin_dt,
        use_abs=False,
    )

    plot_leading_eigenvectors_by_time(
        leading_vecs,
        time_indices,
        output_dir / "leading_eigenvectors_abs",
        label=f"{label}_abs",
        bin_dt=bin_dt,
        use_abs=True,
    )


    plot_eigenvector_contact_sheet(
        leading_vecs,
        time_indices,
        output_dir / f"{label}_leading_eigenvectors_signed_contact_sheet_20x4.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
        use_abs=False,
        shared_scale=True,
    )

    plot_eigenvector_contact_sheet(
        leading_vecs,
        time_indices,
        output_dir / f"{label}_leading_eigenvectors_abs_contact_sheet_20x4.png",
        nrows=20,
        ncols=4,
        bin_dt=bin_dt,
        use_abs=True,
        shared_scale=True,
    )


    plot_leading_eigenvector_colormap(
        leading_vecs,
        time_indices,
        output_dir / f"{label}_leading_eigenvector_colormap_abs.png",
        label=f"{label}: Absolute leading eigenvector coefficients",
        bin_dt=bin_dt,
        use_abs=True,
    )

    plot_leading_eigenvector_colormap(
        leading_vecs,
        time_indices,
        output_dir / f"{label}_leading_eigenvector_colormap_signed.png",
        label=f"{label}: Signed leading eigenvector coefficients",
        bin_dt=bin_dt,
        use_abs=False,
    )

    return eigvals_all, eigvecs_all, leading_vecs


