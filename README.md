# Computational Imaging - Homeworks

This repository contains the homework assignments and codebase for the **Computational Imaging** course. 

## Project Structure

- `computational-imaging/`: Contains the main environment setup and dependencies managed via `uv`.
- `IPPy/`: Custom Inverse Problems Python library (`operators`, `solvers`, `metrics`) used throughout the course to implement Model-Based Reconstructions.
- `hm1/`: Directory dedicated to **Homework 1: Model-Based Reconstructions**, including the problem descriptions, the Python script (`homework1.py`), the Jupyter Notebook (`homework1.ipynb`), and the required `assets/`.

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

To get started, follow these steps to initialize the environment:

1. Ensure `uv` is installed on your system.
2. Navigate to the root directory and synchronize the environment:
   ```bash
   uv sync
   ```
3. To run the Jupyter Notebooks or scripts, use `uv run`:
   ```bash
   uv run jupyter notebook
   ```

## Covered Topics

- **Homework 1:** Inverse Problems, Regularization (Tikhonov, Total Variation, Total-p Variation), and iterative solvers (CGLS, SGP, Chambolle-Pock) applied to Denoising, Deblurring, Super-Resolution, and CT Reconstruction.
