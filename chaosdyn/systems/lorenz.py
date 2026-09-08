import numpy as np

from .base import DynamicalSystem


class Lorenz(DynamicalSystem):
    """
    The Lorenz system.

    The Lorenz equations are:

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Parameters
    ----------
    sigma:
        Prandtl number.
    rho:
        Rayleigh number.
    beta:
        Geometric factor.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    @property
    def dimension(self) -> int:
        return 3

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        return np.array([dx, dy, dz], dtype=float)