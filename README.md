# Euler Lagrange Saddle Point Finder
This script does not implement any known saddle point search schemes such as the Dimer method or Nudged Elastic Band (NEB) method. It employs a unique approach based on the Euler-Lagrange equation to find the MEP directly by minimizing the action integral.

# Minimum Energy Path Finder Using the Euler-Lagrange Equation

This repository contains a Python script to compute the **Minimum Energy Path (MEP)** between two points on a 2D potential energy surface. Using principles from classical mechanics, specifically the **Euler-Lagrange equation**, this method finds the path of least action through a saddle point.

Unlike traditional methods (e.g., Nudged Elastic Band or Dimer), this approach directly minimizes the action integral, avoiding artificial forces or constraints.

---

## Table of Contents
- [Overview](#overview)
- [Theory](#theory)
  - [Potential Energy Surface](#potential-energy-surface)
  - [Lagrangian and Action](#lagrangian-and-action)
  - [Euler-Lagrange Equation](#euler-lagrange-equation)
  - [Saddle Point Constraints](#saddle-point-constraints)
- [Methodology](#methodology)
  - [Discretization and Spline Interpolation](#discretization-and-spline-interpolation)
  - [Optimization Problem](#optimization-problem)
  - [Equations](#equations)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Running the Script](#running-the-script)
  - [Example](#example)
- [Advantages Over NEB and Dimer Methods](#advantages-over-neb-and-dimer-methods)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The script allows users to:
1. Define their own potential energy function or use a default.
2. Compute the MEP by minimizing the action integral.
3. Ensure the path passes through a saddle point, where the gradient of the potential energy is zero, and the Hessian has both positive and negative eigenvalues.

---

## Theory

### Potential Energy Surface
The **Potential Energy Surface (PES)** represents the energy of a system as a function of its spatial coordinates. In two dimensions, it is expressed as $V(x, y)$. Finding the MEP on a PES is crucial for understanding reaction pathways, transition states, and energy barriers.

### Lagrangian and Action
The **Lagrangian** is defined as the difference between kinetic and potential energy:
$$
L = T - V
$$
For this problem:
- **Kinetic Energy**: $T = \frac{1}{2}\left(\frac{dy}{dx}\right)^2$
- **Potential Energy**: $V(x, y)$

Thus, the Lagrangian becomes:
$$
L(x, y, y') = \frac{1}{2}\left(\frac{dy}{dx}\right)^2 + V(x, y)
$$

The **action** $S$ is the integral of the Lagrangian:
$$
S = \int_{x_0}^{x_f} L(x, y, y') \, dx
$$

### Euler-Lagrange Equation
The path $y(x)$ that minimizes $S$ is obtained by solving the **Euler-Lagrange equation**:
$$
\frac{d}{dx}\left(\frac{\partial L}{\partial y'}\right) - \frac{\partial L}{\partial y} = 0
$$
Substituting $L(x, y, y')$, this simplifies to:
$$
\frac{d^2y}{dx^2} - \frac{\partial V}{\partial y} = 0
$$

### Saddle Point Constraints
A **saddle point** satisfies:
1. $\nabla V(x_s, y_s) = 0$ (gradient is zero).
2. Determinant of the Hessian matrix $H$ is negative ($\text{det}(H) < 0$).

---

## Methodology

### Discretization and Spline Interpolation
1. **Discretize** the path between start $(x_0, y_0)$ and end $(x_f, y_f)$.
2. **Initial Guess** for $y$-coordinates (including saddle point) is provided.
3. Use **spline interpolation** to ensure smoothness and differentiability of $y(x)$.

### Optimization Problem
Minimize the action $S$ while enforcing:
- Gradient Constraint: $\nabla V(x_s, y_s) = (0, 0)$
- Hessian Determinant Constraint: $\text{det}(H) < 0$

### Equations
- **Lagrangian**: $L(x, y, y') = \frac{1}{2}\left(\frac{dy}{dx}\right)^2 + V(x, y)$
- **Action**: $S = \int_{x_0}^{x_f} L(x, y, y') \, dx$
- **Euler-Lagrange Equation**: $\frac{d^2y}{dx^2} - \frac{\partial V}{\partial y} = 0$

---

## Usage

### Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- SymPy

Install dependencies:
```bash
pip install numpy scipy matplotlib sympy
git clone https://github.com/faridf/mep-euler-lagrange.git
cd mep-euler-lagrange
python mep_finder.py
