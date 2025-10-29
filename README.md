# Learning-Based Control via Predictive Safety Filters

This repository contains the code used to generate the numerical example in the paper  
**"Learning-Based Control via Predictive Safety Filters"** 

The experiment is an inverted pendulum with:
- hard state and input constraints,
- a moving obstacle that must be avoided,
- a **Predictive Safety Filter (PSF)** that guarantees safety and convergence,
- a learned **performance-boosting policy** (the "actor") that tries to improve performance while remaining safe.

Key idea:
- The learned policy does **not** directly control the plant.
- Instead, it proposes:
  1. a desired input \(u_L\),
  2. a scheduling knob \(\rho_t\) that sets how aggressively the PSF enforces Lyapunov decrease.
- The PSF takes these proposals, solves a short-horizon constrained optimal control problem, and returns a **safe input** that can actually be applied.
- This guarantees stability and constraint satisfaction at all times, even while learning.

This repository reproduces:
- closed-loop trajectories,
- the paper plots (pendulum angle vs. time, cost \(J^\star\)),
<p align="center">
  <img src="figures/figure1_theta_error_overlay.png" width="320"/>
</p>
<p align="center">
  <img src="figures/figure3_jstar.png" width="320"/>
</p>
- an animation of the pendulum avoiding the obstacle.

<p align="center">
  <img src="figures/pendulum_single.gif" width="320"/>
</p>

---

## Table of Contents
1. [Repository Structure](#repository-structure)  
2. [Core Components](#core-components)  
3. [Installation](#installation)  
4. [Quickstart: Train and Log a Run](#quickstart-train-and-log-a-run)  
5. [Generate the Paper Figures](#generate-the-paper-figures)  
6. [Generate the Pendulum GIF](#generate-the-pendulum-gif)  
7. [Technical Notes](#technical-notes)  
8. [Citation](#citation)  
9. [License](#license)

---

## Repository Structure

```text
.
├── run.py
├── plot_results.py
├── create_gifs.py
├── pendulum_env.py
├── single_pendulum_sys.py
├── mad_controller.py
├── PSF.py
├── obstacles.py
├── loss_function.py
├── plot_functions.py
├── results/
│   ├── <RUN_FOLDER_NAME>/
│   │   ├── training.log
│   │   ├── model_best.pth
│   │   ├── model_latest.pth
│   │   ├── rollout_*.npz        # logged trajectories, costs, etc.
│   ├── plots/
│   │   ├── theta_closed_loop.pdf
│   │   ├── cost_J.pdf
│   └── gifs/
│       ├── pendulum_single.gif
└── figures/
    ├── theta_closed_loop.pdf    # copy from results/plots/
    ├── cost_J.pdf               # copy from results/plots/
    ├── pendulum.gif             # copy from results/gifs/
