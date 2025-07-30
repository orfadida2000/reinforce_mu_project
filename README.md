# Reinforce μ Project

This project implements and visualizes reinforcement learning experiments using the REINFORCE algorithm.   
It includes tools for generating learning curves, tracking convergence behaviors, and analyzing the impact of
hyperparameters like learning rate (η).

## 📚 Overview

- **Core Implementation**: The REINFORCE algorithm is implemented in the **[reinforce.training](reinforce/training.py)**
  module.
- **Plotting Utilities**: The **[reinforce.plotting](reinforce/plotting.py)** module provides reusable plotting
  functions
  for visualizing learning
  curves and success rates.
- **Constants Management**: The **[exercise.scripts.assignment_constants](exercise/scripts/assignment_constants.py)**
  module defines constants used across experiments, which can be easily modified to adjust simulation parameters.
- **Documentation**: Detailed docstrings and a structured README for easy navigation and understanding, including a
  comprehensive PDF documentation generated using Sphinx (**[reinforce_mu_project.pdf](reinforce_mu_project.pdf)**).
- **Type Annotations**: The codebase uses type annotations for better clarity and maintainability.
- **Modular Design**: The project is structured to allow easy extension and testing of different components.
- **Easy Setup**: Simple setup instructions with a virtual environment and dependency management
  via **[requirements.txt](requirements.txt)**.
- **Results Storage**: Results of the REINFORCE simulations are saved in a specified directory, which can be customized.
- **Testing**: Basic test cases to ensure functionality, with a focus on the REINFORCE algorithm's behavior.

## 📁 Project Structure

```
reinforce_mu_project/
├── reinforce/
│   ├── __init__.py                # Package init
│   ├── plotting.py                # Plotting utilities for visualizing trajectories and performance
│   ├── training.py                # Core REINFORCE training logic
│   └── utilities.py               # Helper functions (e.g. sampling, reward computation)
│
├── exercise/
│   ├── __init__.py                # Marks 'exercise' as a Python package
│   └── scripts/
│       ├── __init__.py            # Package init
│       ├── assignment_constants.py # Constant parameters used in experiments
│       └── main.py                # Entry script for running experiments
│
├── LICENSE
├── README.md                      # Project documentation (you are here)
├── .gitignore
├── reinforce_mu_project.pdf       # Auto-generated Sphinx PDF documentation
└── requirements.txt               # Python dependencies
```

## 🛠 Setup

### 1. Create Virtual Environment

#### Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

From the project root (**[reinforce_mu_project/](.)**), run:

```bash
python -m exercise.scripts.main
```

This will execute the main REINFORCE simulation pipeline defined in **[main.py](exercise/scripts/main.py)**.

## 📈 Results

The results of the REINFORCE simulations will be saved in the `exercise/plots/` directory.   
This behavior can be customized in the **[assignment_constants.py](exercise/scripts/assignment_constants.py)** file, by
modifying the `save_directories_dict` dictionary.

## 🧪 Features

- Simulates the REINFORCE algorithm across multiple learning rates.
- Computes success rates based on convergence to a target μ.
- Plots learning curves and statistics.

## 📝 Constants

The **[assignment_constants.py](exercise/scripts/assignment_constants.py)** file stores default constants such as:

- Target μ value
- Learning rate schedules
- Simulation parameters

## 📊 Plotting

The **[plotting](reinforce/plotting.py)** module provides reusable plotting utilities for:

- μ trajectory visualization
- Success rate analysis
- Comparison of learning rates

## 📄 License

MIT License.
See **[LICENSE](LICENSE)** for details.

## 👤 Author

- **Name:** Or Fadida
- **Email:** [orfadida@mail.tau.ac.il](mailto:orfadida@mail.tau.ac.il)
- **GitHub:** [orfadida2000](https://github.com/orfadida2000)
