from typing import *

import sympy as sp
import scipy as sc
import numpy as np




def get_cartesian_coordinates(solver_result: sc.integrate._ivp.ivp.OdeResult,
                              length_upper: float,
                              length_lower: float):
    def angle_to_xy(angle: float, distance: float):
        return np.array([distance * np.sin(angle), -distance * np.cos(angle)])

        # Extract the solver's results

    phi_upper, _, phi_lower, __ = solver_result.y

    times = solver_result.t
    upper_pendulum = np.array([angle_to_xy(angle=phi, distance=length_upper).tolist() for phi in phi_upper])
    lower_pendulum = np.array(
        [(angle_to_xy(angle=phi_lower[i], distance=length_lower) + upper_pendulum[i]).tolist() for i in
         range(len(phi_lower))])

    return upper_pendulum, lower_pendulum, times


def get_energies(solver_result: sc.integrate._ivp.ivp.OdeResult,
                 mass_upper: float, mass_lower: float,
                 length_upper: float, length_lower: float,
                 gravity: float):
    t = sp.symbols('t')  # time
    m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g')  # masses, lengths, gravitational acceleration
    phi1, phi2 = sp.symbols('phi1 phi2', cls=sp.Function)  # angles as functions of time

    # Define the angle functions and their derivatives
    phi1, phi2 = phi1(t), phi2(t)
    phi1_dot, phi2_dot = sp.diff(phi1, t), sp.diff(phi2, t)

    # Define the Lagrangian
    U1 = m1 * g * l1 * (1 - sp.cos(phi1))
    U2 = m2 * g * (l1 * (1 - sp.cos(phi1)) + l2 * (1 - sp.cos(phi2)))
    U = U1 + U2

    T1 = sp.Rational(1, 2) * m1 * (l1 * phi1_dot) ** 2
    T2 = sp.Rational(1, 2) * m2 * (
            l1 ** 2 * phi1_dot ** 2 + l2 ** 2 * phi2_dot ** 2 + 2 * l1 * l2 * phi1_dot * phi2_dot * sp.cos(phi1 - phi2))
    T = T1 + T2

    # Lambdify the energy expressions
    T_func = sp.lambdify([m1, m2, l1, l2, phi1, phi2, phi1_dot, phi2_dot], T, 'numpy')
    U_func = sp.lambdify([m1, m2, l1, l2, phi1, phi2, g], U, 'numpy')

    # Extract the solver's results
    phi_1_values, phi_1_dot_values, phi_2_values, phi_2_dot_values = solver_result.y

    # Calculate energies
    kinetic_energies = T_func(mass_upper, mass_lower, length_upper, length_lower, phi_1_values, phi_2_values,
                              phi_1_dot_values, phi_2_dot_values)
    potential_energies = U_func(mass_upper, mass_lower, length_upper, length_lower, phi_1_values, phi_2_values, gravity)

    return kinetic_energies, potential_energies


def downsample_data(upper_pendulum, lower_pendulum, times, n):
    """
    Downsamples the input data by keeping only every n-th value.

    Parameters:
    upper_pendulum (array): Array of upper pendulum positions.
    lower_pendulum (array): Array of lower pendulum positions.
    times (array): Array of time points.
    n (int): Integer specifying the downsampling rate.

    Returns:
    tuple: Downsampled arrays of upper_pendulum, lower_pendulum, and times.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    return upper_pendulum[::n], lower_pendulum[::n], times[::n]
