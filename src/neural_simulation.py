# neural_simulation.py
#
# NEST (NEural Simulation Technology) implementation
# to generate an Asynchronous Irregular (AI) regime
# using Brunel network with delta synapses where
# the (excitatory and inhibitory) spike trains
# from the individual neurons approximate
# a Poisson process

from pathlib import Path
import neural_simulation_utils as n_util


# read default parameters from the config file
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "data" / "config.json"

# load the run configation from .json file
config = n_util.SimulationConfig(CONFIG_FILE)

# each run's data files are added to the .txt file below to analyze later
OUT = BASE_DIR / config.output_dir
OUT.mkdir(parents=True, exist_ok=True)

PREFIX_FILE = "_simulation_prefixes_to_analyze.txt"


def main():
    ''' NEST (Neural Simulation Technology) implementation
        to generate an Asynchronous Irregular (AI) regime
        using Brunel network with delta synapses where
        the spike trains from the individual neurons approximate
        a Poisson process
        '''


    print(f"NEST AI regime simulation with {config.synapse_type} synapses and network seed {config.network_seed}.\n")

    # run the simulation on the excitatory/inhibitory recurrent network
    # current version doesn't return anything,
    # earlier version returned the spike trains
    # results = n_util.sim_recurrent_ei_network(...
    n_util.sim_recurrent_ei_network(
        config=config,
        output_dir=OUT,
        prefix_file=PREFIX_FILE,
        number_of_runs=config.num_sample_runs,
    )


if __name__ == "__main__":
    main()