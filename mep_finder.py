import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
import sympy as sp  # Import sympy

# Define symbolic variables
x_sym, y_sym = sp.symbols('x y')

# Ask user for potential energy function
print("Enter the potential energy function in terms of x and y.")
print("For example: '-sin(pi * x) * sin(pi * y)'")
user_potential_str = input("Potential energy function [Press Enter for default]:\n")
if not user_potential_str.strip():
    user_potential_str = '-sin(pi * x) * sin(pi * y)'  # Default potential
    print("Using default potential energy function.")

# Parse the potential energy function
try:
    potential_expr = sp.sympify(user_potential_str)
except (sp.SympifyError, SyntaxError):
    print("Invalid potential energy function. Using default potential.")
    potential_expr = sp.sympify('-sin(pi * x) * sin(pi * y)')

# Compute the gradient symbolically
grad_V_sym = [sp.diff(potential_expr, var) for var in (x_sym, y_sym)]

# Compute the Hessian matrix symbolically
hessian_V_sym = [[sp.diff(g, var) for var in (x_sym, y_sym)] for g in grad_V_sym]

# Lambdify numerical functions
potential_energy_func = sp.lambdify((x_sym, y_sym), potential_expr, modules=['numpy'])
gradient_potential_energy_func = sp.lambdify((x_sym, y_sym), grad_V_sym, modules=['numpy'])
hessian_potential_energy_func = sp.lambdify((x_sym, y_sym), hessian_V_sym, modules=['numpy'])

def potential_energy(x, y):
    """Compute the potential energy at (x, y)."""
    return potential_energy_func(x, y)

def gradient_potential_energy(x, y):
    """Compute the gradient of the potential energy at (x, y)."""
    grad = gradient_potential_energy_func(x, y)
    return np.array(grad)

def hessian_potential_energy(x, y):
    """Compute the Hessian matrix of the potential energy at (x, y)."""
    hess = hessian_potential_energy_func(x, y)
    return np.array(hess)

def action(variables, x_start, x_end, y_start, y_end, num_points):
    # Extract variables
    internal_points = variables[:-2]
    x_saddle, y_saddle = variables[-2], variables[-1]

    # Construct x and y arrays
    x = np.linspace(x_start, x_end, num_points)
    y = np.hstack(([y_start], internal_points, [y_end]))

    # Insert saddle point into the path
    saddle_index = np.searchsorted(x, x_saddle)
    x = np.insert(x, saddle_index, x_saddle)
    y = np.insert(y, saddle_index, y_saddle)

    # Fit a spline to y(x)
    spline = UnivariateSpline(x, y, k=5, s=0)
    dy = spline.derivative()(x)

    # Compute the integrand
    integrand = 0.5 * dy**2 - potential_energy(x, y)

    # Integrate using Simpson's rule
    return simps(integrand, x)

def gradient_constraint(variables):
    """Constraint function enforcing zero gradient at the saddle point."""
    x_saddle, y_saddle = variables[-2], variables[-1]
    grad_V = gradient_potential_energy(x_saddle, y_saddle)
    return grad_V

def hessian_determinant_constraint(variables):
    """Constraint function enforcing negative determinant of the Hessian at the saddle point."""
    x_saddle, y_saddle = variables[-2], variables[-1]
    H = hessian_potential_energy(x_saddle, y_saddle)
    det_H = np.linalg.det(H)
    return det_H

def find_minimum_energy_path(x_start, x_end, y_start, y_end, num_points):
    # Small epsilon to ensure variables are within bounds
    epsilon = 1e-6

    # Initial guess for internal points, adjusted to be within bounds
    initial_internal = np.linspace(y_start + epsilon, y_end - epsilon, num_points - 2)

    # Initial guess for saddle point (midpoint), adjusted if necessary
    initial_x_saddle = (x_start + x_end) / 2
    initial_y_saddle = (y_start + y_end) / 2

    # Ensure saddle point initial guess is within bounds
    initial_x_saddle = np.clip(initial_x_saddle, -1.0 + epsilon, 1.0 - epsilon)
    initial_y_saddle = np.clip(initial_y_saddle, -1.0 + epsilon, 1.0 - epsilon)

    initial_guess = np.hstack((initial_internal, initial_x_saddle, initial_y_saddle))
    print("Initial guess min:", np.min(initial_guess))
    print("Initial guess max:", np.max(initial_guess))

    # Bounds for y values and saddle point coordinates
    bounds_internal = [(-1.5, 1.5)] * (num_points - 2)
    bounds_saddle = [(-1.0, 1.0), (-1.0, 1.0)]
    bounds = bounds_internal + bounds_saddle

    # Constraints
    gradient_cons = NonlinearConstraint(gradient_constraint, [0, 0], [0, 0])
    epsilon_det = 1e-6
    hessian_det_cons = NonlinearConstraint(hessian_determinant_constraint, -np.inf, -epsilon_det)

    constraints = [gradient_cons, hessian_det_cons]

    result = minimize(
        action,
        initial_guess,
        args=(x_start, x_end, y_start, y_end, num_points),
        method='trust-constr', # CG  SLSQP L-BFGS-B TNC trust-constr
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'maxiter': 100000, 'finite_diff_rel_step': 1e-2, 'verbose': 2}
    )

    # Extract optimized variables
    internal_points = result.x[:-2]
    x_saddle, y_saddle = result.x[-2], result.x[-1]

    # Reconstruct the path
    x = np.linspace(x_start, x_end, num_points)
    y = np.hstack(([y_start], internal_points, [y_end]))

    # Insert saddle point into the path
    saddle_index = np.searchsorted(x, x_saddle)
    x = np.insert(x, saddle_index, x_saddle)
    y = np.insert(y, saddle_index, y_saddle)

    return x, y, (x_saddle, y_saddle)

def visualize_path(x, y, saddle_point):
    # Create a grid for the potential energy surface
    X_grid = np.linspace(-1, 1, 200)
    Y_grid = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(X_grid, Y_grid)
    Z = potential_energy(X, Y)

    # Compute the potential energy along the path
    Z_path = potential_energy(x, y)

    # Create a contour plot to show potential energy levels
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Potential Energy')

    # Plot the minimum energy path
    plt.plot(x, y, 'r-', linewidth=2, label='Minimum Energy Path')

    # Mark the saddle point
    x_saddle, y_saddle = saddle_point
    plt.plot(x_saddle, y_saddle, 'ko', markersize=8, label='Saddle Point')

    # Labels and title
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Contour Plot of Potential Energy Surface with MEP', fontsize=15)
    plt.legend()

    plt.show()

# Set parameters
x_start, x_end = -0.2, 0.8
y_start, y_end = -0.8, 0.2
num_points = 20  # Adjust as needed

# Find the minimum energy path
x, y, saddle_point = find_minimum_energy_path(x_start, x_end, y_start, y_end, num_points)

# Visualize the path
visualize_path(x, y, saddle_point)

# Print the saddle point coordinates
print(f"Saddle Point Coordinates: x = {saddle_point[0]}, y = {saddle_point[1]}")

# Check gradient and determinant at saddle point
grad_V = gradient_potential_energy(saddle_point[0], saddle_point[1])
H = hessian_potential_energy(saddle_point[0], saddle_point[1])
det_H = np.linalg.det(H)
print("Gradient at saddle point:", grad_V)
print("Determinant of Hessian at saddle point:", det_H)

