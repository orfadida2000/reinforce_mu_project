r"""
This module defines global constants used across the reinforcement learning experiment.
These constants include parameters for the experiment setup.
"""

import numpy as np
from reinforce.utilities import reward_function, asymmetric_reward_function

#: The target value for the reward function.
m = 2

#: The initial mean for the normal distribution used to compute y values.
mu0 = 0

#: Dictionary mapping parts of the assignment to their respective standard deviations.
sigmas_dict = {
	"part1": [0.1],
	"part3": [0.1, 0.5]
	}

#: Dictionary mapping parts of the assignment to their respective etas (learning rates).
etas_dict = {
	"part1": np.logspace(-5, -2, 6, dtype=np.float64).tolist(),
	"part3": [0.001]
	}

#: Dictionary mapping parts of the assignment to their respective reward functions.
reward_funcs_dict = {
	"part1": reward_function,
	"part3": asymmetric_reward_function
	}

#: Dictionary mapping parts of the assignment to their respective directory paths for saving plots.
save_directories_dict = {
	"part1": ["exercise", "plots", "mu_trajectories"],
	"part2": ["exercise", "plots", "success_rate_vs_eta"],
	"part3": ["exercise", "plots", "reward_vs_y_with_median"]
	}

#: Number of steps to run in each REINFORCE simulation.
n_steps = 10000

#: Number of repetitions for each simulation to average results.
n_reps = 200

#: Indices for each step in the simulation, used for plotting.
step_indices = np.arange(n_steps)

#: Tolerance for checking convergence in the REINFORCE algorithm.
tolerance = 0.1

#: Colors for plotting final median values resulted from different sigma values on the reward vs. y plot.
median_colors = ["red", "green"]

#: File extension for saving plots.
extension = 'png'

#: DPI (dots per inch) for saving plots, affecting the resolution of saved figures.
dpi = 300
