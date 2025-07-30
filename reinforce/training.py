r"""
This module implements training-related logic for updating the distribution parameter $\mu$.
It supports different reward structures and learning dynamics, and is designed to work
with pluggable reward functions and gradients.
"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray


def run_single_run(
		m: float,
		mu0: float,
		n_steps: int,
		eta: float,
		y_sampler: Callable[[float], float],
		reward_func: Callable[[float, float], float],
		g_func: Callable[[float, float], float],
		g_tag_func: Callable[[float, float], float]
		) -> NDArray[np.float64]:
	r"""
	Runs a single REINFORCE simulation and returns the $\mu$ trajectory.
	
	Parameters:
		m (float): Target value.
		mu0 (float): Initial mu value.
		n_steps (int): Number of steps to run the simulation.
		eta (float): Learning rate.
		y_sampler (Callable[[float], float]): Function to sample y given mu.
		reward_func (Callable[[float, float], float]): Function to compute reward given y and m.
		g_func (Callable[[float, float], float]): Function to compute g(y, mu).
		g_tag_func (Callable[[float, float], float]): Function to compute the derivative of g with respect to mu.
		
	Returns:
		NDArray[np.float64]: Array of mu values over the steps (trajectory).
	"""
	mu = mu0
	mu_trajectory = np.zeros(n_steps)
	for t in range(n_steps):
		y = y_sampler(mu)
		r = reward_func(y, m)
		deriv_lan_g = g_tag_func(y, mu) / g_func(y, mu)
		mu += eta * r * deriv_lan_g
		mu_trajectory[t] = mu
	return mu_trajectory


def run_multiple_repetitions(
		m: float,
		mu0: float,
		n_steps: int,
		eta: float,
		n_reps: int,
		y_sampler: Callable[[float], float],
		reward_func: Callable[[float, float], float],
		g_func: Callable[[float, float], float],
		g_tag_func: Callable[[float, float], float]
		) -> NDArray[np.float64]:
	r"""
	Runs multiple repetitions and returns an array of mu trajectories (n_reps, n_steps) for this eta value.
	
	Parameters:
		m (float): Target value.
		mu0 (float): Initial mu value.
		n_steps (int): Number of steps to run in each simulation.
		eta (float): Learning rate.
		n_reps (int): Number of repetitions to run.
		y_sampler (Callable[[float], float]): Function to sample y given mu.
		reward_func (Callable[[float, float], float]): Function to compute reward given y and m.
		g_func (Callable[[float, float], float]): Function to compute g(y, mu).
		g_tag_func (Callable[[float, float], float]): Function to compute the derivative of g with respect to mu.
		
	Returns:
		NDArray[np.float64]: Array of mu trajectories of shape (n_reps, n_steps).
	"""
	runs = np.zeros((n_reps, n_steps))
	for i in range(n_reps):
		runs[i] = run_single_run(m, mu0, n_steps, eta, y_sampler, reward_func, g_func, g_tag_func)
	return runs
