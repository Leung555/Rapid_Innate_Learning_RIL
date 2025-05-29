# Rapid Innate Learning (RIL)

## Description
This repository contains the code and analysis for the Rapid Innate Learning (RIL) mechanism.

## Contents

*   **`analysis/`**: This directory contains Jupyter notebooks and data files used for analyzing the simulation results.
    *   `*.csv`: Various CSV files containing compliance and simulation data.
*   **`docs/`**: This directory contains supplementary materials.
    *   `supplementary_RIL.pdf`: Supplementary material for the RIL project.
*   **`scenes/`**: This directory contains CoppeliaSim scene files and related scripts.
    *   `script/`: Contains Python scripts for controlling the simulations.
        *   `6_legs.py`: Script for simulations involving a 6-legged robot.
        *   `4_legs.py`: Script for simulations involving a 4-legged robot.
    *   `archived/`: Contains archived scene files.

## Usage

*   **Simulations**: The CoppeliaSim scene files in the `scenes/` directory can be used with the provided Python scripts to run simulations. Ensure that CoppeliaSim is installed and configured correctly.  The scripts use the [zmqremoteapi](scenes/script/4_legs.py) to interface with CoppeliaSim.
*   **Analysis**: The Jupyter notebooks in the `analysis/` directory can be used to analyze the data generated from the simulations.  These notebooks use `pandas`, `matplotlib`, and `seaborn` for data manipulation and visualization.

## Supplementary Materials
The supplementary material of this project can be found in the [doc/supplementary_RIL.pdf](docs/supplementary_RIL.pdf).
