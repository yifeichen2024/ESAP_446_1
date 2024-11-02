import numpy as np
from numpy.typing import NDArray
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import math
import finite


# from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import deque
from functools import cache
from typing import Any, Callable, Optional, cast

import numpy as np
import scipy.sparse.linalg as spla  # type: ignore
from numpy.typing import NDArray
from scipy import sparse  # type: ignore
from scipy.special import factorial  # type: ignore

from farray import apply_matrix, axslice
from finite import Difference



class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.dt = dt
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f


class ImplicitTimestepper(Timestepper):

    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt*self.f(self.u)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        k = [np.zeros_like(self.u) for _ in range(self.stages)]
        for i in range(self.stages):
            u_temp = np.copy(self.u)
            for j in range(i):
                u_temp += dt * self.a[i, j] * k[j]
            k[i] = self.f(u_temp)
        return self.u + dt * sum(self.b[i] * k[i] for i in range(self.stages))


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.previous_fs = []

    def _step(self, dt):
        if len(self.previous_fs) < self.steps - 1:
            # Use Forward Euler for the first few steps
            self.previous_fs.append(self.f(self.u))
            return self.u + dt * self.f(self.u)
        else:
            # Use Adams-Bashforth method
            self.previous_fs.append(self.f(self.u))
            if len(self.previous_fs) > self.steps:
                self.previous_fs.pop(0)
            coefficients = self._get_coefficients()
            return self.u + dt * sum(coeff * f_val for coeff, f_val in zip(coefficients, reversed(self.previous_fs)))

    def _get_coefficients(self):
        # Coefficients for Adams-Bashforth methods of different orders
        if self.steps == 1:
            return [1]
        elif self.steps == 2:
            return [3/2, -1/2]
        elif self.steps == 3:
            return [23/12, -16/12, 5/12]
        elif self.steps == 4:
            return [55/24, -59/24, 37/24, -9/24]
        elif self.steps == 5:
            return [1901/720, -1387/360, 109/30, -637/360, 251/720]
        elif self.steps == 6:
            return [4277/1440, -2641/480, 4991/720, -3649/720, 959/480, -95/288]
        else:
            raise ValueError("Adams-Bashforth method only implemented for up to 6 steps.")



class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.u)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.L.matrix
            self.RHS = self.I + dt/2*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.RHS @ self.u)


class BackwardDifferentiationFormula(Timestepper):
    steps: int
    thist: list[float]
    uhist: list[NDArray[np.float64]]

    def __init__(self, u: NDArray[np.float64], L_op: Difference, steps: int):
        super().__init__()
        self.u = u
        self.func = L_op
        self.steps = steps
        self.thist = []
        self.uhist = []

    def _step(self, dt: float) -> NDArray[np.float64]:
        self.thist.append(dt)
        self.uhist.append(self.u)
        steps = min(self.steps, len(self.uhist))
        solve = self._coeff(tuple(self.thist[-steps:]))
        return solve(np.stack(self.uhist[-steps:], axis=1))

    @cache
    def _coeff(
        self, thist: tuple[float, ...]
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        (N,) = self.u.shape
        steps = len(thist)
        x = np.cumsum(np.array((0,) + thist))
        xx = x[-1]
        x /= xx
        coeff = np.zeros((steps + 1,))
        for i in range(steps + 1):
            poly = np.array([1.0])
            for j in range(steps + 1):
                if i != j:
                    poly = np.convolve(poly, np.array([1.0, -x[j]]))
                    poly /= x[i] - x[j]
            poly = poly[:-1] * np.arange(steps, 0, -1)
            coeff[i] = poly @ (x[-1] ** np.arange(steps - 1, -1, -1))
        coeff /= xx
        lu = spla.splu(self.func.matrix - coeff[-1] * sparse.eye(N, N))
        return lambda u: cast(NDArray[np.float64], lu.solve(u @ coeff[:-1]))




# class StateVector:
#     data: NDArray[np.float64]

#     def __init__(self, variables: list[NDArray[np.float64]], axis: int = 0):
#         self.axis = axis
#         var0 = variables[0]
#         shape = list(var0.shape)
#         self.N = shape[axis]
#         shape[axis] *= len(variables)
#         self.shape = tuple(shape)
#         self.data = np.zeros(shape)
#         self.variables = variables
#         self.gather()

#     def gather(self) -> None:
#         for i, var in enumerate(self.variables):
#             np.copyto(  # type: ignore
#                 self.data[axslice(self.axis, i * self.N, (i + 1) * self.N)], var
#             )

#     def scatter(self) -> None:
#         for i, var in enumerate(self.variables):
#             np.copyto(  # type: ignore
#                 var, self.data[axslice(self.axis, i * self.N, (i + 1) * self.N)]
#             )

class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])


class EquationSet(metaclass=ABCMeta):
    X: StateVector
    M: NDArray[np.float64]
    L: NDArray[np.float64]
    F: Optional[Callable[[StateVector], NDArray[np.float64]]]

def RK22(eq_set: EquationSet) -> Multistage:
    a = np.array([[0, 0], [1 / 2, 0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class IMEXTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1


# class IMEXTimestepper(metaclass=ABCMeta):
#     t: float
#     iter: int
#     X: StateVector
#     M: NDArray[np.float64]
#     L: NDArray[np.float64]
#     dt: Optional[float]

#     @abstractmethod
#     def _step(self, dt: float) -> NDArray[np.float64]:
#         pass

#     def __init__(self, eq_set: EquationSet):
#         assert eq_set.F is not None
#         self.t = 0
#         self.iter = 0
#         self.X = eq_set.X
#         self.M = eq_set.M
#         self.L = eq_set.L
#         self.F = eq_set.F
#         self.dt = None

#     def evolve(self, dt: float, time: float) -> None:
#         while self.t < time - 1e-8:
#             self.step(dt)

#     def step(self, dt: float) -> None:
#         self.X.gather()
#         self.X.data = self._step(dt)
#         self.X.scatter()
#         self.t += dt
#         self.iter += 1




class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)
        

# class CNAB(IMEXTimestepper):
#     def _step(self, dt: float) -> NDArray[np.float64]:
#         LHS: Any
#         RHS: NDArray[np.float64]
#         if self.iter == 0:
#             # Euler
#             LHS = self.M + dt * self.L
#             LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")

#             self.FX = self.F(self.X)
#             RHS = cast(NDArray[np.float64], self.M @ self.X.data) + dt * self.FX
#             self.FX_old = self.FX
#             return cast(NDArray[np.float64], LU.solve(RHS))
#         else:
#             if dt != self.dt:
#                 LHS = self.M + dt / 2 * self.L
#                 self.LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")
#             self.dt = dt

#             self.FX = self.F(self.X)
#             RHS = (
#                 cast(NDArray[np.float64], self.M @ self.X.data)
#                 - cast(NDArray[np.float64], 0.5 * dt * self.L @ self.X.data)
#                 + cast(NDArray[np.float64], 3 / 2 * dt * self.FX)
#                 - cast(NDArray[np.float64], 1 / 2 * dt * self.FX_old)
#             )
#             self.FX_old = self.FX
#             return cast(NDArray[np.float64], self.LU.solve(RHS))



class BDFExtrapolate(IMEXTimestepper):
    '''
    HW5 Part1
    Calculates the future value of X, denoted by X^n, using the current value X^{n-1}.
    and past values.

    _coeff(): Coefficient a and b are found via Taylor expansion 
    
    '''
    coeffs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    xhist: list[NDArray[np.float64]]
    fhist: list[NDArray[np.float64]]

    def __init__(self, eq_set: EquationSet, steps: int):
        super().__init__(eq_set)
        self.steps = steps
        self.xhist = []
        self.fhist = []
        for s in range(1, steps + 1):
            if len(self.coeffs) < s:
                a = np.zeros((s + 1,))
                b = np.zeros((s,))
                for i in range(s + 1):
                    poly = np.array([1.0])
                    x1 = i / s
                    for j in range(s + 1):
                        if i != j:
                            x2 = j / s
                            poly = np.convolve(poly, np.array([1.0, -x2]))
                            poly /= x1 - x2
                        if i < s and j == s - 1:
                            b[i] = poly.sum()
                    a[i] = poly[:-1] @ np.arange(s, 0, -1)
                self.coeffs.append((a, b))

    def _step(self, dt: float) -> NDArray[np.float64]:
        self.xhist.append(self.X.data)
        self.fhist.append(self.F(self.X))
        steps = min(self.steps, len(self.xhist))
        solve = self._coeff(dt, steps)
        return solve(
            np.stack(self.xhist[-steps:], axis=1), np.stack(self.fhist[-steps:], axis=1)
        )

    @cache
    def _coeff(
        self, dt: float, steps: int
    ) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
        a, b = self.coeffs[steps - 1]
        a = cast(NDArray[np.float64], a / (steps * dt))
        lu = spla.splu(self.L + a[-1] * self.M)
        return lambda x, f: cast(
            NDArray[np.float64], lu.solve(f @ b - self.M @ (x @ a[:-1]))
        )