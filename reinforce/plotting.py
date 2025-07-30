r"""
This module defines all visualization functions used in the reinforcement learning experiments.
It includes tools to plot reward trajectories, success rates, and convergence behavior
across different learning rates and repetitions.
"""

from typing import List, Callable, Tuple
from matplotlib import figure, axes, pyplot as plt
import os
import numpy as np
from numpy.typing import NDArray
from reinforce.utilities import is_valid_fs_name, is_valid_plt_extension, get_unique_plot_filename

#: Type alias for matplotlib Figure object, used for plotting.
MPLFig = figure.Figure

#: Type alias for matplotlib Axes object, used for plotting.
Axes = axes.Axes


def figure_saving_assertion(filename: str, directory_path_lst: List[str], extension: str, dpi: int) -> None:
	r"""
	Asserts that the provided filename, directory path list, extension, and DPI are valid for saving a figure.
	
	Parameters:
		filename (str): Base name for the file (without extension).
		directory_path_lst (List[str]): List of folder names forming the relative path.
		extension (str): File extension for saving the plot (e.g., 'png', 'pdf').
		dpi (int): Resolution in dots per inch for saving the figure.
	"""
	assert is_valid_fs_name(
			filename), "File base name and directory name must be valid file system names."
	assert isinstance(directory_path_lst, list) and all(
			is_valid_fs_name(directory) for directory in
			directory_path_lst), "directory_path_lst must be a list of valid file system names."
	assert is_valid_plt_extension(
			extension), "Extension must be a valid matplotlib file extension (e.g., 'png', 'pdf')."
	assert isinstance(dpi, int) and dpi > 0, "DPI must be a positive integer."


def save_figure(fig: MPLFig, filename: str, directory_path_lst: List[str], extension: str, dpi: int) -> None:
	r"""
	Saves a Matplotlib figure to a specified path with a unique filename.
	
	Parameters:
		fig (MPLFig): The Matplotlib figure to save.
		filename (str): Base name for the file (without extension).
		directory_path_lst (List[str]): List of folder names forming the relative path.
		extension (str): File extension for saving the plot (e.g., 'png', 'pdf').
		dpi (int): Resolution in dots per inch for saving the figure.
	"""
	assert isinstance(fig, MPLFig), "fig must be a Matplotlib Figure instance."
	figure_saving_assertion(filename, directory_path_lst, extension, dpi)
	if len(directory_path_lst) > 0:
		directory_path = os.path.join(*directory_path_lst)
		os.makedirs(directory_path, exist_ok=True)  # Ensure the plots directory exists
		file_path = os.path.join(directory_path, filename)
	else:
		file_path = filename
	# Save the figure with a unique filename
	final_path = get_unique_plot_filename(file_path, extension)
	fig.savefig(final_path, dpi=dpi)


def create_eta_figure(
		mu_runs: NDArray[np.float64],
		eta: float,
		step_indices: NDArray[np.int64]) -> MPLFig:
	r"""
	Creates a figure plotting:
	- 5 real runs based on rank positions (5th, 25th, 40th, 60th, 90th percentiles)
	- 1 stepwise median (50th percentile across all runs at each step), with proper math formatting.

	Parameters:
		mu_runs (NDArray[np.float64]): Mu trajectories of shape (n_reps, n_steps).
		eta (float): Learning rate value.
		step_indices (NDArray[np.int64]): Step indices for x-axis.

	Returns:
		MPLFig: The Matplotlib figure with six curves (5 real runs + median trajectory).
	"""
	percentiles_real_runs = [5, 25, 40, 60, 90]
	colors = ['blue', 'dodgerblue', 'green', 'orange', 'red']
	labels = [rf'${p}^{{\mathrm{{th}}}}$ percentile run' for p in percentiles_real_runs]
	
	final_values = mu_runs[:, -1]
	sorted_indices = np.argsort(final_values)
	n_reps = len(final_values)
	
	selected_indices = []
	for p in percentiles_real_runs:
		rank_idx = int(p / 100 * (n_reps - 1))
		index = sorted_indices[rank_idx]
		selected_indices.append(index)
	
	fig, ax = plt.subplots(figsize=(12, 7))
	
	# Plot 5 real percentile-based runs
	i = 0
	plot_median = False
	for idx, color, label in zip(selected_indices, colors, labels):
		if percentiles_real_runs[i] > 50 and not plot_median:
			# Stepwise median (not tied to any single run)
			median_curve = np.percentile(mu_runs, 50, axis=0)
			ax.plot(step_indices, median_curve, color='black', linewidth=2,
					label=r'$50^{\mathrm{th}}$ percentile (median)', zorder=10)
			plot_median = True
		ax.plot(step_indices, mu_runs[idx], label=label, color=color, linewidth=1, zorder=1)
		i += 1
	
	ax.set_title(rf'$\mu$ Trajectories Across Training Steps ($\eta$ = {eta:.2e})', fontsize=16)
	ax.set_xlabel('Steps', fontsize=14)
	ax.set_ylabel(r'$\mu$', fontsize=14, rotation=0, labelpad=20)
	ax.legend(fontsize=12)
	fig.canvas.manager.set_window_title(rf'μ Trajectories (η = {eta:.2e})')
	return fig


def create_success_rate_vs_eta_figure(etas: List[float],
									  success_rates: NDArray[np.float64]) -> MPLFig:
	r"""
	Creates a figure plotting the convergence success rate against the learning rate ($\eta$) and returns it.

	Parameters:
		etas (List[float]): Learning rates ($\eta$) for which success rates were computed.
		success_rates (NDArray[np.float64]): Success rates corresponding to each eta.

	Returns:
		MPLFig: The Matplotlib figure showing success rate vs. learning rate.
	"""
	fig, ax = plt.subplots(figsize=(12, 7))
	ax.semilogx(etas, success_rates * 100, marker='o')
	ax.set_xlabel(r'Learning rate ($\eta$)', fontsize=14)
	ax.set_ylabel('Success rate (%)', fontsize=14)
	ax.set_title('Convergence Success Rate vs. Learning Rate', fontsize=16)
	ax.grid(True, which='both', linestyle='--', linewidth=0.5)
	fig.canvas.manager.set_window_title('Success Rate vs. Learning Rate')
	return fig


def create_reward_vs_y_figure(reward_func: Callable[[float, float], float], m: float, y_min: float,
							  y_max: float) -> Tuple[MPLFig, Axes]:
	r"""
	Creates a figure plotting the reward function against y values within a specified range.

	Parameters:
		reward_func (Callable[[float, float], float]): Function to compute the reward given y and m.
		m (float): Target value.
		y_min (float): Minimum value of y for the plot.
		y_max (float): Maximum value of y for the plot.

	Returns:
		Tuple[MPLFig, Axes]: The Matplotlib figure and axes containing the plot.
	"""
	y_values = np.linspace(y_min, y_max, 1000)
	rewards = [reward_func(y, m) for y in y_values]  # Compute rewards for each y value
	
	fig, ax = plt.subplots(figsize=(12, 7))
	ax.plot(y_values, rewards, label='Reward Function', color='blue')
	ax.set_xlabel('y', fontsize=14)
	ax.set_ylabel('Reward', fontsize=14)
	ax.set_xlim(y_min, y_max)
	ax.set_title('Reward Function vs. y', fontsize=16)
	ax.grid(True)
	ax.legend(fontsize=12)
	fig.canvas.manager.set_window_title('Reward Function vs. y')
	return fig, ax
