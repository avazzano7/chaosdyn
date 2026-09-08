from abc import ABC, abstractmethod

import numpy as np


class DynamicalSystem(ABC):
    """Base class for dynamical systems"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of state variables in the system."""
        raise NotImplementedError

    @abstractmethod
    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the system state.
        
        Parameters
        ----------
        t:
            Current time.
        state:
            Current state verctor.
            
        Returns
        -------
        numpy.ndarray
            Derivative of the state vector.
        """
        raise NotImplementedError