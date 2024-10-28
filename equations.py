from timesteppers_hw5 import StateVector
from scipy import sparse
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Union, cast

from finite import (
    Difference,
    DifferenceNonUniformGrid,
    DifferenceUniformGrid,
    NonUniformPeriodicGrid,
    UniformPeriodicGrid,
)
from timesteppers import CrankNicolson, EquationSet, StateVector


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0*X.data


# class SoundWave:

#     def __init__(self, u, p, d, rho0, gammap0):
#         pass


# class ReactionDiffusion:
    
#     def __init__(self, c, d2, c_target, D):
#         pass

class SoundWave(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        p: NDArray[np.float64],
        d: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
        rho0: Union[float, NDArray[np.float64]],
        gammap0: Union[float, NDArray[np.float64]],
    ):
        (N,) = u.shape
        I = sparse.eye(N, N)  # noqa: E741
        Z = sparse.csr_matrix((N, N))
        self.X = StateVector([u, p])
        self.M = sparse.bmat([[rho0 * sparse.csc_array(I), Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [gammap0 * sparse.csc_array(d.matrix), Z]])
        self.F = lambda X: np.zeros(X.data.shape)


class ReactionDiffusion(EquationSet):
    def __init__(
        self,
        c: NDArray[np.float64],
        d2: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
        c_target: Union[float, NDArray[np.float64]],
        D: float,
    ):
        (N,) = c.shape
        self.X = StateVector([c])
        self.M = sparse.eye(N, N)
        self.L = -D * d2.matrix
        self.F = lambda X: X.data * (c_target - X.data)