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
---

## Overview
The script allows users to:
1. Define their own potential energy function or use a default.
2. Compute the MEP by minimizing the action integral.
3. Ensure the path passes through a saddle point, where the gradient of the potential energy is zero, and the Hessian has both positive and negative eigenvalues.

---

## Theory

### Potential Energy Surface
The **Potential Energy Surface (PES)** represents the energy of a system as a function of its spatial coordinates. In two dimensions, it is expressed as `V(x, y)`. Finding the MEP on a PES is crucial for understanding reaction pathways, transition states, and energy barriers.

### Lagrangian and Action
The **Lagrangian** is defined as the difference between kinetic and potential energy:

    L = T - V

For this problem:
- **Kinetic Energy**: `T = 0.5 * (dy/dx)^2`
- **Potential Energy**: `V(x, y)`

Thus, the Lagrangian becomes:

    L(x, y, y') = 0.5 * (dy/dx)^2 + V(x, y)

The **action** `S` is the integral of the Lagrangian:

    S = ∫[x0 to xf] L(x, y, y') dx

### Euler-Lagrange Equation
The path `y(x)` that minimizes `S` is obtained by solving the **Euler-Lagrange equation**:

    d/dx (∂L/∂y') - ∂L/∂y = 0

Substituting `L(x, y, y')`, this simplifies to:

    d^2y/dx^2 - ∂V/∂y = 0

### Saddle Point Constraints
A **saddle point** satisfies:
1. Gradient is zero:

       ∇V(xs, ys) = (∂V/∂x, ∂V/∂y) = (0, 0)

2. Determinant of the Hessian matrix is negative:

       det(H) = (∂^2V/∂x^2)(∂^2V/∂y^2) - (∂^2V/∂x∂y)^2 < 0

---

## Methodology

### Discretization and Spline Interpolation
1. **Discretize** the path between start `(x0, y0)` and end `(xf, yf)`.
2. **Initial Guess** for `y`-coordinates (including saddle point) is provided.
3. Use **spline interpolation** to ensure smoothness and differentiability of `y(x)`.

### Optimization Problem
Minimize the action `S` while enforcing:
- Gradient Constraint: `∇V(xs, ys) = (0, 0)`
- Hessian Determinant Constraint: `det(H) < 0`

### Equations
- **Lagrangian**: `L(x, y, y') = 0.5 * (dy/dx)^2 + V(x, y)`
- **Action**: `S = ∫[x0 to xf] L(x, y, y') dx`
- **Euler-Lagrange Equation**: `d^2y/dx^2 - ∂V/∂y = 0`

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
git clone https://github.com/yourusername/mep-euler-lagrange.git
cd mep-euler-lagrange
python mep_finder.py
```


When prompted, enter the potential energy function or press Enter to use the default:
```bash
V(x, y) = -sin(πx) * sin(πy)
```


### Example
Running the script with the default potential:

```bash
python mep_finder.py
```


Output:

A contour plot of the PES with the MEP and saddle point.
Console details: saddle point coordinates, gradient, and Hessian determinant.

## Advantages Over NEB and Dimer Methods

Direct Minimization: No artificial forces or constraints.
No Spring Forces: Avoids constructs like NEB's spring forces.
Flexibility: Adapts to any user-defined potential energy function.
Saddle Point Identification: Explicitly searches for saddle points.

## Limitations

Computational Cost: High for complex potentials or many discretization points.
Dimensionality: Current implementation is limited to 2D.
Initial Guess Sensitivity: Optimization success depends on initial guess.

## Contributing
Contributions are welcome! Feel free to submit pull requests or open issues.

![Figure_1](https://github.com/user-attachments/assets/6f597267-bd30-4baf-be1b-e025392ab66f)



Number of iterations: 368, function evaluations: 13545, CG iterations: 1037, optimality: 4.83e+00, constraint violation: 6.99e-21, execution time:  2.9 s.
Saddle Point Coordinates: x = -7.077349822562493e-22, y = -7.077350130900838e-22
Gradient at saddle point: [6.9850646e-21 6.9850643e-21]
Determinant of Hessian at saddle point: -97.40909103400243


