from __future__ import annotations

import numpy as np

from chaosdyn.systems.base import DynamicalSystem


def rk4_step(
    system: DynamicalSystem,
    t: float,
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Advance a dynamical system by one RK4 timestep.

    Parameters
    ----------
    system:
        Dynamical system being integrated.
    t:
        Current time.
    state:
        Current state vector.
    dt:
        Timestep.

    Returns
    -------
    numpy.ndarray
        State after one timestep.
    """
    k1 = system.derivatives(t, state)
    k2 = system.derivatives(t + dt / 2, state + dt * k1 / 2)
    k3 = system.derivatives(t + dt / 2, state + dt * k2 / 2)
    k4 = system.derivatives(t + dt, state + dt * k3)

    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve(
    system: DynamicalSystem,
    initial_state: np.ndarray | list[float],
    t_span: tuple[float, float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate a dynamical system using the classical RK4 method.

    Parameters
    ----------
    system:
        Dynamical system to integrate.
    initial_state:
        Initial state vector.
    t_span:
        Tuple containing start and end times.
    dt:
        Timestep.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Time values and corresponding state trajectory.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    t_start, t_end = t_span

    if t_end <= t_start:
        raise ValueError("t_span must have t_end > t_start.")

    state = np.asarray(initial_state, dtype=float)

    if state.shape != (system.dimension,):
        raise ValueError(
            f"Expected initial_state with shape "
            f"({system.dimension},), got {state.shape}."
        )

    n_steps = int(np.ceil((t_end - t_start) / dt))

    times = np.empty(n_steps + 1)
    trajectory = np.empty((n_steps + 1, system.dimension))

    times[0] = t_start
    trajectory[0] = state

    t = t_start

    for i in range(1, n_steps + 1):
        step = min(dt, t_end - t)

        state = rk4_step(system, t, state, step)

        t += step

        times[i] = t
        trajectory[i] = state

    return times, trajectory