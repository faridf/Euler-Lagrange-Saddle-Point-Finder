import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import trapz
from scipy.interpolate import UnivariateSpline
import sympy as sp

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

# Set parameters
x_start, x_end = -0.2, 0.8
y_start, y_end = -0.8, 0.2
epsilon = 1e-6  # Small epsilon to avoid bounds issues

def action(variables, num_points):
    """Compute the action for the given path."""
    num_internal_points = num_points - 2
    x_internal = variables[:num_internal_points]
    y_internal = variables[num_internal_points:2*num_internal_points]
    # s_saddle is the last variable (not used in action but in constraints)
    s_saddle = variables[-1]
    
    # Control points including start and end
    x_control = np.hstack(([x_start], x_internal, [x_end]))
    y_control = np.hstack(([y_start], y_internal, [y_end]))
    
    # Parameter s for control points and dense evaluation
    s_control = np.linspace(0, 1, num_points)
    s_dense = np.linspace(0, 1, 200)  # Increase for smoother path
    
    # Create spline functions
    spline_x = UnivariateSpline(s_control, x_control, k=3, s=0)
    spline_y = UnivariateSpline(s_control, y_control, k=3, s=0)
    
    # Evaluate the path and its derivatives
    x = spline_x(s_dense)
    y = spline_y(s_dense)
    dx_ds = spline_x.derivative()(s_dense)
    dy_ds = spline_y.derivative()(s_dense)
    
    # Compute the integrand of the action (subtract potential energy)
    integrand = 0.5 * (dx_ds**2 + dy_ds**2) - potential_energy(x, y)
    
    # Integrate over s to compute the action
    action_value = trapz(integrand, s_dense)
    return action_value

def gradient_constraint(variables):
    """Constraint function enforcing zero gradient at the saddle point."""
    num_internal_points = num_points - 2
    x_internal = variables[:num_internal_points]
    y_internal = variables[num_internal_points:2*num_internal_points]
    s_saddle = variables[-1]
    
    # Control points including start and end
    x_control = np.hstack(([x_start], x_internal, [x_end]))
    y_control = np.hstack(([y_start], y_internal, [y_end]))
    s_control = np.linspace(0, 1, num_points)
    
    # Create spline functions
    spline_x = UnivariateSpline(s_control, x_control, k=3, s=0)
    spline_y = UnivariateSpline(s_control, y_control, k=3, s=0)
    
    # Evaluate the saddle point
    x_saddle = spline_x(s_saddle)
    y_saddle = spline_y(s_saddle)
    
    grad_V = gradient_potential_energy(x_saddle, y_saddle)
    return grad_V

def hessian_determinant_constraint(variables):
    """Constraint function enforcing negative determinant of the Hessian at the saddle point."""
    num_internal_points = num_points - 2
    x_internal = variables[:num_internal_points]
    y_internal = variables[num_internal_points:2*num_internal_points]
    s_saddle = variables[-1]
    
    # Control points including start and end
    x_control = np.hstack(([x_start], x_internal, [x_end]))
    y_control = np.hstack(([y_start], y_internal, [y_end]))
    s_control = np.linspace(0, 1, num_points)
    
    # Create spline functions
    spline_x = UnivariateSpline(s_control, x_control, k=3, s=0)
    spline_y = UnivariateSpline(s_control, y_control, k=3, s=0)
    
    # Evaluate the saddle point
    x_saddle = spline_x(s_saddle)
    y_saddle = spline_y(s_saddle)
    
    H = hessian_potential_energy(x_saddle, y_saddle)
    det_H = np.linalg.det(H)
    return det_H

def find_minimum_energy_path(num_points):
    """Find the Minimum Energy Path (MEP) between start and end points."""
    num_internal_points = num_points - 2
    
    # Initial guesses for internal control points
    initial_x_internal = np.linspace(x_start + epsilon, x_end - epsilon, num_internal_points)
    initial_y_internal = np.linspace(y_start + epsilon, y_end - epsilon, num_internal_points)
    
    # Initial guess for s_saddle (midpoint)
    initial_s_saddle = 0.5
    
    # Combine into a single array
    initial_guess = np.hstack((initial_x_internal, initial_y_internal, initial_s_saddle))
    
    # Define bounds for the control points and s_saddle
    bounds_x = [(-1.5, 1.5)] * num_internal_points
    bounds_y = [(-1.5, 1.5)] * num_internal_points
    bounds_s_saddle = [(0.0, 1.0)]  # s_saddle between 0 and 1
    bounds = bounds_x + bounds_y + bounds_s_saddle
    
    # Constraints with a small tolerance
    tol = 1e-6
    gradient_cons = NonlinearConstraint(gradient_constraint, [-tol, -tol], [tol, tol])
    hessian_det_cons = NonlinearConstraint(hessian_determinant_constraint, -np.inf, -tol)
    
    constraints = [gradient_cons, hessian_det_cons]
    
    # Perform the optimization to minimize the action
    result = minimize(
        action,
        initial_guess,
        args=(num_points,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'maxiter': 10000}
    )
    
    # Extract optimized control points and s_saddle
    x_internal = result.x[:num_internal_points]
    y_internal = result.x[num_internal_points:2*num_internal_points]
    s_saddle = result.x[-1]
    
    # Reconstruct the full path including start and end points
    x_control = np.hstack(([x_start], x_internal, [x_end]))
    y_control = np.hstack(([y_start], y_internal, [y_end]))
    
    # Create spline functions to evaluate the saddle point
    s_control = np.linspace(0, 1, num_points)
    spline_x = UnivariateSpline(s_control, x_control, k=3, s=0)
    spline_y = UnivariateSpline(s_control, y_control, k=3, s=0)
    x_saddle = spline_x(s_saddle)
    y_saddle = spline_y(s_saddle)
    
    return x_control, y_control, (x_saddle, y_saddle)

def visualize_path(x_control, y_control, saddle_point):
    """Visualize the potential energy surface and the Minimum Energy Path (MEP)."""
    # Create spline functions for plotting
    s_dense = np.linspace(0, 1, 200)
    s_control = np.linspace(0, 1, len(x_control))
    spline_x = UnivariateSpline(s_control, x_control, k=3, s=0)
    spline_y = UnivariateSpline(s_control, y_control, k=3, s=0)
    x = spline_x(s_dense)
    y = spline_y(s_dense)
    
    # Create a grid for the potential energy surface
    X_grid = np.linspace(-1, 1, 200)
    Y_grid = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(X_grid, Y_grid)
    Z = potential_energy(X, Y)

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

if __name__ == "__main__":
    num_points = 20  # Adjust as needed

    # Find the minimum energy path
    x_control, y_control, saddle_point = find_minimum_energy_path(num_points)
    
    # Visualize the path
    visualize_path(x_control, y_control, saddle_point)
    
    # Print the saddle point coordinates
    print(f"Saddle Point Coordinates: x = {saddle_point[0]}, y = {saddle_point[1]}")
    
    # Check gradient and determinant at saddle point
    grad_V = gradient_potential_energy(saddle_point[0], saddle_point[1])
    H = hessian_potential_energy(saddle_point[0], saddle_point[1])
    det_H = np.linalg.det(H)
    print("Gradient at saddle point:", grad_V)
    print("Determinant of Hessian at saddle point:", det_H)
