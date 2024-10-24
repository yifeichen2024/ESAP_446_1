import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import math
import finite


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


class BackwardDifferentiationFormula(ImplicitTimestepper):
    def __init__(self, u, L, steps):
        super().__init__(u, L)
        self.steps = steps  # Order of the BDF method
        self.history_u = []  # Stores previous u values
        self.history_dt = []  # Stores previous timestep sizes
        self.coefficients = None  # BDF coefficients
        self.L_matrix = L.matrix  # The operator L as a matrix
        self.LU = None  # LU decomposition of the LHS matrix
        self.I = sparse.eye(self.L_matrix.shape[0], format='csc')  # Identity matrix
        self.dt = None  # Previous timestep size

        
    # def compute_coefficients(self, dt_list):
    #     s = len(dt_list)
    #     if not all(dt == dt_list[0] for dt in dt_list):
    #         raise ValueError("Variable timesteps not supported in this implementation.")
    
    #     if s == 1:
    #         self.coefficients = np.array([1, -1])
    #     elif s == 2:
    #         self.coefficients = np.array([3/2, -2, 1/2])
    #     elif s == 3:
    #         self.coefficients = np.array([11/6, -3, 3/2, -1/3])
    #     elif s == 4:
    #         self.coefficients = np.array([25/12, -4, 3, -4/3, 1/4])
    #     elif s == 5:
    #         self.coefficients = np.array([137/60, -5, 5/2, -5/3, 5/12, -1/5])
    #     elif s == 6:
    #         self.coefficients = np.array([147/60, -6, 5, -10/3, 5/2, -6/5, 1/6])
    #     else:
    #         raise ValueError("Unsupported number of steps for BDF method.")

    def compute_coefficients(self, dt_list):
        """
        Computes BDF coefficients for variable timesteps.
        """
        s = len(dt_list)
        k = np.arange(1, s+1)
        dt_ratios = np.array(dt_list[-s:]) / dt_list[-1]
        gamma = np.ones(s+1)
        for j in range(1, s+1):
            gamma[j] = gamma[j-1] * dt_ratios[-j]

        # Build the Vandermonde matrix
        A = np.zeros((s+1, s+1))
        for i in range(s+1):
            A[i, :] = np.power(-k, i)
        b = np.zeros(s+1)
        b[1] = -1  # Corresponds to the derivative term

        # Solve for the coefficients
        self.coefficients = np.linalg.solve(A, b)

    def runge_kutta_step(self, dt):
        # 4th-order Runge–Kutta method
        k1 = dt * self.L_matrix.dot(self.u)
        k2 = dt * self.L_matrix.dot(self.u + 0.5 * k1)
        k3 = dt * self.L_matrix.dot(self.u + 0.5 * k2)
        k4 = dt * self.L_matrix.dot(self.u + k3)
        return self.u + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def CrankNicolson(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.L.matrix
            self.RHS = self.I + dt/2*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.RHS @ self.u)

    def trapezoidal_step(self, dt):
        """
        Trapezoidal method for the initial steps.
        """
        if self.dt != dt or not hasattr(self, 'LU'):
            self.LHS = self.I - dt / 2 * self.L_matrix
            self.RHS_matrix = self.I + dt / 2 * self.L_matrix
            self.LU = spla.splu(self.LHS)
            self.dt = dt  # Update stored dt

        # Compute RHS
        RHS = self.RHS_matrix.dot(self.u)

        # Solve the linear system
        return self.LU.solve(RHS)
    
    def _step(self, dt):
        # Update the timestep history
        self.history_dt.append(dt)
        if len(self.history_dt) > self.steps:
            self.history_dt.pop(0)

        # Update the solution history
        self.history_u.append(self.u.copy())
        if len(self.history_u) > self.steps:
            self.history_u.pop(0)

        # Check if we have enough steps to use the full BDF method
        if len(self.history_u) < self.steps + 1:
            self.u = self.trapezoidal_step(dt)
            # # Use Crank–Nicolson method for initial steps
            # if self.dt != dt or not hasattr(self, 'LU'):
            #     self.LHS = self.I - dt / 2 * self.L_matrix
            #     self.RHS_matrix = self.I + dt / 2 * self.L_matrix
            #     self.LU = spla.splu(self.LHS)
            #     self.dt = dt  # Update stored dt

            # # Compute RHS
            # RHS = self.RHS_matrix.dot(self.u)

            # Solve the linear system
            # self.u = self.LU.solve(RHS)
        else:
            # We have enough steps; use BDF method
            # Compute coefficients
            dt_list = self.history_dt[-self.steps:]
            self.compute_coefficients(dt_list)
            coeffs = self.coefficients  # Array of length steps + 1

            # Assemble LHS: coeffs[0] * I - L_matrix
            LHS = coeffs[0] * self.I - self.L_matrix

            # Assemble RHS
            RHS = np.zeros_like(self.u)
            for i in range(1, self.steps + 1):
                RHS -= coeffs[i] * self.history_u[-(i + 1)]

            # Solve the linear system
            self.LU = spla.cg(LHS)
            self.u = self.LU.solve(RHS)

        # Return the updated solution
        return self.u
    
