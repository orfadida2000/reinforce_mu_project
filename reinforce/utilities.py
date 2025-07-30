r"""
This module provides utility functions for the reinforcement learning setup.
It includes tools for generating samples, defining reward functions,
validating names, defining reward functions and organizing file paths.
"""

import os
import re
from typing import Callable, Tuple

from matplotlib import pyplot as plt
import numpy as np

#: Regular expression pattern for validating basic filesystem names (alphanumeric, underscores, hyphens, spaces).
BASIC_SAFE_PATTERN = re.compile(r'^[A-Za-z0-9_\- .]+$')

#: Type alias for a tuple of three functions used in the REINFORCE algorithm (y_sampler, g_func, g_tag_func).
REINFORCE_FUNCS = Tuple[
	Callable[[float], float], Callable[[float, float], float], Callable[[float, float], float]]


def get_unique_plot_filename(base_name: str, extension: str = "png") -> str:
	r"""
	Generates a unique filename by appending a number if the base name already exists.
	
	Parameters:
		base_name (str): Base name for the file (relative or absolute path without extension).
		extension (str): File extension (default is "png").
		
	Returns:
		str: Unique filename with the specified base name and extension.
	"""
	
	filename = f"{base_name}.{extension}"
	if not os.path.exists(filename):
		return filename
	i = 1
	while True:
		filename = f"{base_name}({i}).{extension}"
		if not os.path.exists(filename):
			return filename
		i += 1


def is_valid_fs_name(name: str) -> bool:
	r"""
	Checks if name is a valid filesystem name (not empty, alphanumeric, underscores, hyphens, and spaces).
	
	Parameters:
		name (str): Name to validate.
	
	Returns:
		bool: True if the name is valid, False otherwise.
	"""
	if not isinstance(name, str):
		return False
	if not bool(BASIC_SAFE_PATTERN.match(name)):
		return False
	return name == name.strip() and not name.startswith('.') and not name.endswith('.')


def is_valid_plt_extension(extension: str) -> bool:
	r"""
	Checks if the given file extension is supported by Matplotlib for saving figures.
	
	Parameters:
		extension (str): File extension to check (e.g., 'png', 'pdf').
		
	Returns:
		bool: True if the extension is valid, False otherwise.
	"""
	valid_plt_extensions = plt.gcf().canvas.get_supported_filetypes().keys()
	if not isinstance(extension, str):
		return False
	if extension.lower() in valid_plt_extensions:
		return True
	return False


def sample_y_normal(mu: float, sigma: float) -> float:
	r"""
	Samples a value y from a normal distribution centered at mu with standard deviation sigma.

	Parameters:
		mu (float): Mean of the normal distribution.
		sigma (float): Standard deviation of the normal distribution.

	Returns:
		float: Sampled value y.
	"""
	return np.random.normal(mu, sigma)


def reward_function(y: float, m: float) -> float:
	r"""
	Computes the reward for a given value y and target m using squared distance.

	Parameters:
		y (float): The sampled value.
		m (float): The target value.

	Returns:
		float: The reward value, negative squared distance.
	"""
	return -(m - y) ** 2


def scaled_reward_function(y: float, m: float) -> float:
	r"""
	Computes the scaled reward (scaled by 2) for a given value y and target m using squared distance.

	Parameters:
		y (float): The sampled value.
		m (float): The target value.

	Returns:
		float: The reward value, two times the negative squared distance.
	"""
	return -2 * (m - y) ** 2


def asymmetric_reward_function(y: float, m: float) -> float:
	r"""
	Computes the asymmetric reward for a given value y and target m using squared distance.

	Parameters:
		y (float): The sampled value.
		m (float): The target value.

	Returns:
		float: The reward value, two times the negative squared distance if y <= m, otherwise the negative squared distance.
	"""
	if y <= m:
		return scaled_reward_function(y, m)
	else:
		return reward_function(y, m)


def g_normal(y: float, mu: float, sigma: float) -> float:
	r"""
	Probability density of y given mu and sigma (normal distribution).
	
	Parameters:
		y (float): The value for which to compute the density.
		mu (float): Mean of the normal distribution.
		sigma (float): Standard deviation of the normal distribution.
	
	Returns:
		float: Probability density of y given mu and sigma.
	"""
	coef = 1 / (np.sqrt(2 * np.pi) * sigma)
	exponent = -((y - mu) ** 2) / (2 * sigma ** 2)
	return coef * np.exp(exponent)


def g_tag_normal(y: float, mu: float, sigma: float) -> float:
	r"""
	Calculates the derivative of g(y | $\mu$, $\sigma$) with respect to $\mu$ at a given y and returns it.
	
	Parameters:
		y (float): The value for which to compute the gradient.
		mu (float): Mean of the normal distribution.
		sigma (float): Standard deviation of the normal distribution.
	
	Returns:
		float: The derivative of the probability density function with respect to mu.
	"""
	return g_normal(y, mu, sigma) * (y - mu) / (sigma ** 2)


def generate_reinforce_normal_funcs(sigma: float) -> REINFORCE_FUNCS:
	r"""
	Generates the REINFORCE functions for a normal distribution with given standard deviation.
	More specifically, it generates the sampling function for y, the probability density function g(y | $\mu$, $\sigma$), and the derivative of g with respect to $\mu$.
	
	Parameters:
		sigma (float): Standard deviation of the normal distribution.
		
	Returns:
		REINFORCE_FUNCS: A tuple containing:
			- y_sampler: Function to sample y given mu.
			- g_func: Function to compute g(y | $\mu$, $\sigma$).
			- g_tag_func: Function to compute the derivative of g with respect to $\mu$.
	"""
	y_sampler = lambda mu: sample_y_normal(mu, sigma)
	g_func = lambda y, mu: g_normal(y, mu, sigma)
	g_tag_func = lambda y, mu: g_tag_normal(y, mu, sigma)
	return y_sampler, g_func, g_tag_func
