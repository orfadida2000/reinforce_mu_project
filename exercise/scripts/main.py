r"""
Main script for running the $\mu$-convergence reinforcement learning experiment.

This script configures the experiment using the constants file, initializes the
learning loop, and produces relevant plots and statistics.
Designed to be executed directly from the command line.
"""

from numpy.typing import NDArray
from typing import Callable, Dict, Union
import exercise.scripts.assignment_constants as constants
from reinforce.plotting import (create_eta_figure, create_success_rate_vs_eta_figure,
								create_reward_vs_y_figure, save_figure)
from reinforce.training import run_multiple_repetitions
from reinforce.utilities import generate_reinforce_normal_funcs

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

#: Type alias for a generic float type.
Float = Union[float, np.floating]


def compute_success_rate(mu_trajectories: NDArray[np.float64],
						 m: float,
						 tolerance: float) -> float:
	r"""
	Computes the success rate (percentage of runs) where the final mu converges within [m - tolerance, m + tolerance].

	Parameters:
		mu_trajectories (NDArray[np.float64]): Array of mu trajectories with shape (n_reps, n_steps).
		m (float): Target value.
		tolerance (float): Acceptable deviation from target m for successful convergence.

	Returns:
		float: Success rate as a fraction of runs that converged successfully.
	"""
	final_mus = mu_trajectories[:, -1]
	successful = np.abs(final_mus - m) <= tolerance
	return successful.mean()


def compute_success_rates_from_trajectories(eta_mu_pairs: List[Tuple[float, NDArray[np.float64]]],
											m: float,
											tolerance: float) -> NDArray[np.float64]:
	r"""
	Computes success rates from precomputed mu trajectories per learning rate.

	Parameters:
		eta_mu_pairs (List[Tuple[float, NDArray[np.float64]]]):
			List of tuples. Each tuple contains:
				- float : learning rate eta
				- np.ndarray : mu trajectories with shape (n_reps, n_steps).
		m (float): Target value.
		tolerance (float): Acceptable deviation from target m for successful convergence.

	Returns:
		NDArray[np.float64]: Array of success rates, ordered according to eta_mu_pairs.
	"""
	success_rates = np.zeros(len(eta_mu_pairs))
	for idx, (eta, mu_traj) in enumerate(eta_mu_pairs):
		success_rates[idx] = compute_success_rate(mu_traj, m, tolerance)
	return success_rates


def run_part_one(y_sampler: Callable[[float], float],
				 reward_func: Callable[[float, float], float],
				 g_func: Callable[[float, float], float],
				 g_tag_func: Callable[[float, float], float],
				 m: float,
				 mu0: float,
				 n_steps: int,
				 n_reps: int,
				 etas: List[float],
				 step_indices: NDArray[np.int64],
				 save_directories: List[str],
				 extension: str,
				 dpi: int) -> List[Tuple[float, NDArray[np.float64]]]:
	r"""
	Runs part one of the experiment: compute mu trajectories for different etas and save the figures.
	
	Parameters:
		y_sampler (Callable[[float], float]): Function to sample y given mu.
		reward_func (Callable[[float, float], float]): Function to compute reward given y and m.
		g_func (Callable[[float, float], float]): Function to compute g(y, mu).
		g_tag_func (Callable[[float, float], float]): Function to compute the derivative of g with respect to mu.
		m (float): Target value.
		mu0 (float): Initial mu value.
		n_steps (int): Number of steps.
		n_reps (int): Number of repetitions.
		etas (List[float]): List of learning rates.
		step_indices (NDArray[np.int64]): Step indices for x-axis.
		save_directories (List[str]): List of folders forming the path to save plots.
		extension (str): File extension for saving plots (e.g., 'png', 'pdf').
		dpi (int): Dots per inch for saved plots.
	
	Returns:
		List[Tuple[float, NDArray[np.float64]]]: List of tuples containing eta and corresponding mu trajectories.
	"""
	print("Start running part one of the experiment...")
	eta_mu_pairs = []
	for eta in etas:
		print(f"Running for eta = {eta:.2e}")
		mu_runs = run_multiple_repetitions(m, mu0, n_steps, eta, n_reps, y_sampler, reward_func, g_func,
										   g_tag_func)
		fig = create_eta_figure(mu_runs, eta, step_indices)
		filename = f'mu_trajectory_eta_{eta:.2e}'
		save_figure(fig, filename, save_directories, extension, dpi)
		plt.show()
		plt.close(fig)
		eta_mu_pairs.append((eta, mu_runs))
	return eta_mu_pairs


def run_part_two(eta_mu_pairs: List[Tuple[float, NDArray[np.float64]]],
				 m: float,
				 tolerance: float,
				 save_directories: List[str],
				 extension: str,
				 dpi: int) -> None:
	r"""
	Runs part two of the experiment: computes success rates from mu trajectories and plot/save success rate vs eta.

	Parameters:
		eta_mu_pairs (List[Tuple[float, NDArray[np.float64]]]): List of tuples containing eta and corresponding mu trajectories.
		m (float): Target value.
		tolerance (float): Acceptable deviation from target m for successful convergence.
		save_directories (List[str]): List of folders forming the path to save the figure.
		extension (str): File extension for saving the figure (e.g., 'png', 'pdf').
		dpi (int): Dots per inch for saved plots.
	"""
	print("Start running part two of the experiment...")
	success_rates = compute_success_rates_from_trajectories(eta_mu_pairs, m, tolerance)
	etas = [eta for eta, _ in eta_mu_pairs]
	fig = create_success_rate_vs_eta_figure(etas, success_rates)
	filename = 'success_rate_vs_eta'
	save_figure(fig, filename, save_directories, extension, dpi)
	plt.show()
	plt.close(fig)


def run_part_three(y_sampler_lst: List[Callable[[float], float]],
				   reward_func: Callable[[Float, Float], Float],
				   g_func_lst: List[Callable[[float, float], float]],
				   g_tag_func_lst: List[Callable[[float, float], float]],
				   sigmas: List[float],
				   median_colors: List[str],
				   m: float,
				   mu0: float,
				   n_steps: int,
				   n_reps: int,
				   eta: float,
				   save_directories: List[str],
				   extension: str,
				   dpi: int) -> None:
	r"""
	Runs part three of the experiment: simulates REINFORCE with different sigmas and plots reward vs y with median points.
	
	Parameters:
		y_sampler_lst (List[Callable[[float], float]]): List of functions to sample y given mu for different sigmas.
		reward_func (Callable[[Float, Float], Float]): Function to compute reward given y and m.
		g_func_lst (List[Callable[[float, float], float]]): List of functions to compute g(y, mu) for different sigmas.
		g_tag_func_lst (List[Callable[[float, float], float]]): List of functions to compute the derivative of g with respect to mu for different sigmas.
		sigmas (List[float]): List of standard deviations for sampling y.
		median_colors (List[str]): List of colors for plotting median points.
		m (float): Target value.
		mu0 (float): Initial mu value.
		n_steps (int): Number of steps.
		n_reps (int): Number of repetitions.
		eta (float): Learning rate.
		save_directories (List[str]): List of folders forming the path to save plots.
		extension (str): File extension for saving plots (e.g., 'png', 'pdf').
		dpi (int): Dots per inch for saved plots.
	"""
	print("Start running part three of the experiment...")
	fig, ax = create_reward_vs_y_figure(reward_func, m, -1.0, 5.0)
	
	for sigma, y_sampler, g_func, g_tag_func, color in zip(sigmas, y_sampler_lst, g_func_lst, g_tag_func_lst,
														   median_colors):
		print(f"Running for sigma = {sigma:.2f}")
		mu_runs = run_multiple_repetitions(m, mu0, n_steps, eta, n_reps, y_sampler, reward_func, g_func,
										   g_tag_func)
		final_values = mu_runs[:, -1]
		median_value = np.percentile(final_values, 50)
		reward_at_median = reward_func(median_value, m)
		ax.scatter(median_value, reward_at_median, color=color, s=80,
				   label=rf'Median $\mu$ = {median_value:.2f} ($\sigma$ = {sigma:.2f})')
	
	ax.legend(fontsize=12)
	filename = 'reward_vs_y'
	save_figure(fig, filename, save_directories, extension, dpi)
	plt.show()
	plt.close(fig)


def run_full_experiment(reward_funcs_dict: Dict[str, Callable[[float, float], float]],
						m: float,
						tolerance: float,
						median_colors: List[str],
						mu0: float,
						sigmas_dict: Dict[str, List[float]],
						n_steps: int,
						n_reps: int,
						etas_dict: Dict[str, List[float]],
						step_indices: NDArray[np.int64],
						save_directories_dict: Dict[str, List[str]],
						extension: str,
						dpi: int = 300) -> None:
	r"""
	Runs full experiment across different etas and saves the corresponding figures.

	Parameters:
		reward_funcs_dict (Dict[str, Callable[[float, float], float]]): Dictionary with keys 'part1', 'part3', each containing a reward function.
		m (float): Target value.
		tolerance (float): Acceptable deviation from target m for successful convergence.
		median_colors (List[str]): List of colors for plotting median points in part three.
		mu0 (float): Initial mu value.
		sigmas_dict (Dict[str, List[float]]): Dictionary with keys 'part1', 'part3', each containing a list of floats representing the standard deviation for sampling y.
		n_steps (int): Number of steps.
		n_reps (int): Number of repetitions.
		etas_dict (Dict[str, List[float]]): Dictionary with keys 'part1', 'part3', each containing a list of learning rates for different parts of the experiment.
		step_indices (NDArray[np.int64]): Step indices for x-axis.
		save_directories_dict (Dict[str, List[str]]): Dictionary with keys 'part1', 'part2', 'part3', each containing a list of folders forming the path to save plots.
		extension (str): File extension for saving plots (e.g., 'png', 'pdf').
		dpi (int): Dots per inch for saved plots (default is 300).
	"""
	
	print("Start running the full experiment...")
	
	# Run part one: compute mu trajectories for different etas and save the figures
	sigma1 = sigmas_dict["part1"][0]  # Use the first sigma for part one
	y_sampler1, g_func1, g_tag_func1 = generate_reinforce_normal_funcs(sigma1)
	eta_mus_pairs = run_part_one(y_sampler1, reward_funcs_dict["part1"], g_func1, g_tag_func1, m, mu0,
								 n_steps,
								 n_reps,
								 etas_dict["part1"],
								 step_indices,
								 save_directories_dict["part1"], extension, dpi)
	
	# Run part two: compute success rates and save the success rate vs eta figure
	run_part_two(eta_mus_pairs, m, tolerance, save_directories_dict["part2"], extension, dpi)
	
	# Run part three: simulate REINFORCE with different sigmas and plot/save reward vs y with median points
	y_sampler_lst = []
	g_func_lst = []
	g_tag_func_lst = []
	eta = etas_dict["part3"][0]  # Use the first eta for part three
	sigmas = sigmas_dict["part3"]
	for sigma in sigmas:
		y_sampler, g_func, g_tag_func = generate_reinforce_normal_funcs(sigma)
		y_sampler_lst.append(y_sampler)
		g_func_lst.append(g_func)
		g_tag_func_lst.append(g_tag_func)
	run_part_three(y_sampler_lst, reward_funcs_dict["part3"], g_func_lst, g_tag_func_lst, sigmas,
				   median_colors, m, mu0, n_steps, n_reps, eta, save_directories_dict["part3"], extension,
				   dpi)
	
	print("Full experiment completed successfully!")


if __name__ == "__main__":
	run_full_experiment(constants.reward_funcs_dict, constants.m, constants.tolerance,
						constants.median_colors, constants.mu0,
						constants.sigmas_dict, constants.n_steps,
						constants.n_reps, constants.etas_dict, constants.step_indices,
						constants.save_directories_dict,
						constants.extension, constants.dpi)
